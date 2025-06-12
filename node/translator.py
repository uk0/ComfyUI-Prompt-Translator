import os
import re
import json
import time
import requests


import re

def maybe_fix_encoding(value: str) -> str:
    """
    尝试修复常见的“UTF-8 原本是中文，但被误当作 Latin-1 解码”导致的乱码。
    如果不是这种情况，就原样返回。
    常见特征是出现 'æ', 'è', 'é' 等莫名符号组合。
    """
    try:
        # 将字符串按 latin-1 回编码成 bytes，再用 utf-8 解码
        repaired = value.encode('latin-1', errors='strict').decode('utf-8', errors='strict')
        return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        # 如果出错，说明并不是这种形式的错误编码
        return value

def llm_extract_fields(text: str, fields: list) -> dict:
    """
    从大模型返回的文本中提取指定字段的内容，支持：
      - 三引号 Markdown 代码块（```xxx ... ```）的完整捕获
      - 普通带引号字符串（含转义、中文、多行）
      - 未加引号的简单值、数字、布尔、null 等
      - 中文乱码的自动检测与尝试修复

    注意事项：
      1. 如果内容中本身含有更多复杂嵌套（如三引号里再出现三引号），仍有可能干扰匹配；
      2. 如果大模型输出中完全无对应字段，则不会出现在结果里；
      3. 如果字段多次出现，以第一次匹配到的为准（或可根据需求修改为最后一次匹配）；
      4. 若确认文本确实是合法 JSON，优先使用 `json.loads` 更可靠；
      5. 这里所有正则匹配都启用了 DOTALL (让 '.' 匹配换行) 与 UNICODE (支持中文) 等选项。

    Args:
        text (str): 大模型返回的原始文本。
        fields (list): 需要提取的字段列表，例如 ["title", "content", "code"]。

    Returns:
        dict: {字段名: 提取到的值(字符串或数字/布尔/None)}

    示例：
        text = \"""
        {
          "title": "Golang Kafka 消费者代码示例",
          "content": \"多行文字 \\n 含中文\",
          "snippet": ```go
        package main
        ...
        ```
        }
        \"""
        result = llm_extract_fields(text, ["title", "content", "snippet"])
        # result -> {
        #   "title": "Golang Kafka 消费者代码示例",
        #   "content": "多行文字 \n 含中文",
        #   "snippet": "package main\n..."
        # }
    """

    result = {}

    # 依次匹配所需字段
    for field in fields:
        # 如果已经提取到该字段，就跳过（避免重复操作）
        if field in result:
            continue

        # -- 1. 代码块模式：用三引号包裹  (```lang\n ... ```)
        #    可能大模型会输出: "fieldName": ```go\n...```
        #    正则思路：先匹配 "fieldName": ，然后找 ```，可选语言标识，最后一直到下一组 ```
        code_pattern = rf'"{re.escape(field)}"\s*:\s*```[\w]*[\r\n]*([\s\S]*?)```'
        match = re.search(code_pattern, text, re.DOTALL | re.UNICODE)
        if match:
            value = match.group(1)
            # 去掉首尾多余空行或空格
            value = value.strip('\r\n ')
            # 尝试修复乱码
            value = maybe_fix_encoding(value)
            result[field] = value
            continue

        # -- 2. 带引号字符串：支持转义字符、多行
        #    形如: "fieldName": "...."
        str_pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(str_pattern, text, re.DOTALL | re.UNICODE)
        if match:
            value = match.group(1)
            # 先把 \u4e2d\u6587 之类的转义序列转换回中文
            value = bytes(value, "utf-8").decode("unicode_escape", errors='replace')
            # 再尝试修复“æ…”类乱码
            value = maybe_fix_encoding(value)
            result[field] = value
            continue

        # -- 3. 数字 (整数或浮点)
        #    形如: "fieldName": 123  或  3.14
        num_pattern = rf'"{re.escape(field)}"\s*:\s*([+-]?\d+(?:\.\d+)?)'
        match = re.search(num_pattern, text, re.UNICODE)
        if match:
            num_str = match.group(1)
            # 判断是否小数
            if '.' in num_str:
                try:
                    num_val = float(num_str)
                    result[field] = num_val
                    continue
                except ValueError:
                    pass  # 万一无法转换，就不处理
            else:
                try:
                    int_val = int(num_str)
                    result[field] = int_val
                    continue
                except ValueError:
                    pass

        # -- 4. 布尔值
        bool_pattern = rf'"{re.escape(field)}"\s*:\s*(true|false)'
        match = re.search(bool_pattern, text, re.IGNORECASE | re.UNICODE)
        if match:
            bool_str = match.group(1).lower()
            result[field] = (bool_str == 'true')
            continue

        # -- 5. null
        null_pattern = rf'"{re.escape(field)}"\s*:\s*null'
        match = re.search(null_pattern, text, re.IGNORECASE | re.UNICODE)
        if match:
            result[field] = None
            continue

        # -- 6. 未加引号的字符串或其他类型
        #    如: "fieldName": someValueWithoutQuotes
        #    这里直到下一个逗号、换行或右花括号才算结束
        unquoted_pattern = rf'"{re.escape(field)}"\s*:\s*([^\s,}}]+)'
        match = re.search(unquoted_pattern, text, re.UNICODE)
        if match:
            value = match.group(1).strip()
            value = maybe_fix_encoding(value)
            result[field] = value
            continue

    return result

def translate(text, max_retries=3, backoff_factor=1.0):
    """
    调用大模型接口将中文描述转为 Flux Dev AI 生成图像的 英文正/负 启动词（prompt）。
    如果返回的 HTTP 状态码不是 2xx，将最多重试 max_retries 次，且每次重试前等待 backoff_factor * (2 ** (retry_idx-1)) 秒。
    """
    api_url = os.getenv("BIGMODEL_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    api_key = os.getenv("BIGMODEL_API_KEY", "xxxxxxxxxxxxxxxx")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    system_prompt = """
    你是一名专业的提示工程师，负责生成适用于 Flux Dev AI 基于 CLIP 的图像生成参数。

    当收到用户的中文描述时，请按照以下要求输出**唯一**的 JSON 对象，不要添加任何多余说明：
    
    1. **positive_prompt**：一个简洁的、用逗号分隔的英文关键词和短短语列表（最长约 77 个 token），精确描述要包含的内容（物体、构图、风格、光照、色彩、情绪、视角、细节），以优化 CLIP 嵌入——只用清晰的名词和形容词，不用填充词。  
    2. **negative_prompt**：一个用逗号分隔的英文关键词列表，列出需要排除的不良元素或伪影（噪点、扭曲、不想要的物体等），同样针对 CLIP 嵌入优化——只用简单的名词和形容词。  
    3. **num_images**：用户需要生成的图像数量，范围 1–4；默认 1，如果描述中未提及或超出范围则置为 1。  
    4. **steps**：采样步数，范围 15–50；默认 15，如果描述中提及“步骤”且数值有效则使用该值，若小于 15 则置为 15，若大于 50 则置为 50。  
    5. **cfg**：CLIP 引导强度，范围 1.0–15.0；默认 5.0，如果描述中提及“cfg”且数值有效则使用该值，若小于 1.0 则置为 5.0，若大于 15.0 则置为 15.0。
    
    示例输出格式（仅 JSON，不要解释）：
    
    ```json
    {
      "positive_prompt": "sunset over mountain lake, warm golden light, misty atmosphere, high detail, panoramic view",
      "negative_prompt": "noise, blur, distorted faces, text, watermarks",
      "num_images": 2,
      "steps": 30,
      "cfg": 7.5
    }
    """

    payload = {
        "model": "glm-z1-flash",
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
            # 如果不是 2xx，就抛出异常进入重试
            resp.raise_for_status()

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # 1. 移除所有完整的 <think>...</think> 内容（包括标签）
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # 2. 若存在孤立的 </think>，则去掉从文本开头到此标签（包含标签）
            content_cleaned = re.sub(r'^.*?</think>', '', content, flags=re.DOTALL)


            # json_match = re.search(r'\{.*\}', content_cleaned, re.S)
            # if not json_match:
            #     raise ValueError("未找到有效JSON")
            # return json.loads(json_match.group(0))
            print(content_cleaned)
            json_match = llm_extract_fields(content_cleaned,
                                            ["positive_prompt", "negative_prompt", "num_images", "steps", "cfg"])
            return json_match

        except Exception as e:
            last_exception = e
            # 如果已到达最大重试次数，则退出循环
            if attempt == max_retries:
                break
            # 否则等待并重试
            wait = backoff_factor * (2 ** (attempt - 1))
            print(f"第 {attempt} 次调用失败：{e}，{wait:.1f}s 后重试...")
            time.sleep(wait)

    # 最终失败，打印错误并返回空结果
    print("翻译或解析 JSON 最终失败：", last_exception)
    return {"positive_prompt": "", "negative_prompt": "", "num_images": ""}


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fa5]', text))


class PromptTextTranslation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_trans": ("STRING", {"multiline": True, "default": "海边，日出"}),
                "trans_switch": (["enabled", "disabled"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING","INT","INT","FLOAT")
    RETURN_NAMES = ("Positive Prompt", "Negative Prompt","Image Count", "Steps", "CFG")
    FUNCTION = "translation"
    CATEGORY = "utils"

    def translation(self, text_trans, trans_switch):
        if text_trans == "undefined":
            text_trans = ""

        print("tf prompt: ", text_trans)

        if trans_switch == "enabled" and contains_chinese(text_trans):
            result = translate(text_trans)
        else:
            result = {"positive_prompt": text_trans, "negative_prompt": "", "num_images": 1,"steps": 20, "cfg": 1}

        positive = result.get("positive_prompt", "")
        negative = result.get("negative_prompt", "")
        num_images = result.get("num_images", 1)
        steps = result.get("steps", 20)
        cfg = result.get("cfg", 1)

        # 清理逗号和空格
        positive = re.sub(r'\s+,', ',', positive.replace('，', ',').replace('。', ',')).strip()
        negative = re.sub(r'\s+,', ',', negative.replace('，', ',').replace('。', ',')).strip()

        print("Positive Prompt:", positive)
        print("Negative Prompt:", negative)
        print("Num Images:", num_images)
        print("Images steps:", steps)
        print("Images cfg:", cfg)

        return (positive, negative,num_images,steps,cfg)


# 测试示例
if __name__ == "__main__":
    example_prompt = "一个漂亮的女人，站在海边，日出时分，阳光照耀着她的身体，生成四张图片,步骤20，cfg值1.5 3张图片"

    translator = PromptTextTranslation()
    pos, neg ,num_img,steps,cfg= translator.translation(example_prompt, "enabled")
    print("==============================\n")
    print("最终输出的 Positive Prompt:", pos)
    print("最终输出的 Negative Prompt:", neg)
    print("最终输出的 num_img:", num_img)
    print("最终输出的 steps:", steps)
    print("最终输出的 cfg:", cfg)

    example_prompt = "一个漂亮的女人，站在海边，日出时分，阳光照耀着她的身体，生成四张图片,steps 0，cfg值 0"

    translator = PromptTextTranslation()
    pos, neg, num_img, steps, cfg = translator.translation(example_prompt, "enabled")
    print("==============================\n")
    print("最终输出的 Positive Prompt:", pos)
    print("最终输出的 Negative Prompt:", neg)
    print("最终输出的 num_img:", num_img)
    print("最终输出的 steps:", steps)
    print("最终输出的 cfg:", cfg)