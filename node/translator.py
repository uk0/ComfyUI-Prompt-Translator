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
    api_key = os.getenv("BIGMODEL_API_KEY", "xxxxxxxxx")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    system_prompt = """
    You are a specialized prompt engineer optimizing text for CLIP-based image generation in Flux Dev AI.

    When given a Chinese description from the user, generate two CLIP-compatible English prompts:
    
    1. (CLIP Positive Prompt) A concise, comma-separated list of high-value keywords and short phrases (max ~77 tokens) describing exactly what to include: objects, composition, style, lighting, color palette, mood, perspective, and any fine details. Structure it for optimal CLIP embedding — clear nouns and adjectives, no filler words.
    
    2. (CLIP Negative Prompt) A comma-separated list of undesirable elements, artifacts, or styles to exclude — noise, distortions, unwanted objects, etc. Keep it focused on CLIP’s understanding (simple nouns and adjectives).
    
    Output **only** a JSON object with exactly two fields, without any extra text:
    
    ```json
    {
      "positive_prompt": "<your CLIP-optimized positive prompt>",
      "negative_prompt": "<your CLIP-optimized negative prompt>"
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
    return {"positive_prompt": "", "negative_prompt": ""}


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

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("Positive Prompt", "Negative Prompt",)
    FUNCTION = "translation"
    CATEGORY = "utils"

    def translation(self, text_trans, trans_switch):
        if text_trans == "undefined":
            text_trans = ""

        print("输入prompt: ", text_trans)

        if trans_switch == "enabled" and contains_chinese(text_trans):
            result = translate(text_trans)
        else:
            result = {"positive_prompt": text_trans, "negative_prompt": ""}

        positive = result.get("positive_prompt", "")
        negative = result.get("negative_prompt", "")

        # 清理逗号和空格
        positive = re.sub(r'\s+,', ',', positive.replace('，', ',').replace('。', ',')).strip()
        negative = re.sub(r'\s+,', ',', negative.replace('，', ',').replace('。', ',')).strip()

        print("Positive Prompt:", positive)
        print("Negative Prompt:", negative)

        return (positive, negative)


# 测试示例
if __name__ == "__main__":
    example_prompt = "一个漂亮的女人，站在海边，日出时分，阳光照耀着她的身体。"

    translator = PromptTextTranslation()
    pos, neg = translator.translation(example_prompt, "enabled")

    print("\n最终输出的Positive Prompt:", pos)
    print("最终输出的Negative Prompt:", neg)