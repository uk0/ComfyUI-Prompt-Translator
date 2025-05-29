import os
import re
import json
import time
import requests

def translate(text, max_retries=3, backoff_factor=1.0):
    """
    调用大模型接口将中文描述转为 Flux Dev AI 生成图像的 英文正/负 启动词（prompt）。
    如果返回的 HTTP 状态码不是 2xx，将最多重试 max_retries 次，且每次重试前等待 backoff_factor * (2 ** (retry_idx-1)) 秒。
    """
    api_url = os.getenv("BIGMODEL_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    api_key = os.getenv("BIGMODEL_API_KEY", "xxxxxxxxxxxxxxxxxxxxxxx")
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

            # 去除 <think> 标签里的内容
            content_cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.S).strip()
            # 提取 JSON
            json_match = re.search(r'\{.*\}', content_cleaned, re.S)
            if not json_match:
                raise ValueError("未找到有效JSON")

            return json.loads(json_match.group(0))

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

        print("输入prompt: ", text_trans)

        if trans_switch == "enabled" and contains_chinese(text_trans):
            result = translate(text_trans)
        else:
            result = {"positive_prompt": text_trans, "negative_prompt": "", "num_images": 1}

        positive = result.get("positive_prompt", "")
        negative = result.get("negative_prompt", "")
        num_images = result.get("num_images", "")
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
    example_prompt = "一个漂亮的女人，站在海边，日出时分，阳光照耀着她的身体，生成四张图片,步骤20，cfg值1.5"

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