"""阿里云百炼千问模型调用。"""

from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError

from src.config import load_qwen_config
from src.prompts import JUNIOR_PHYSICS_SYSTEM_PROMPT


def answer_question(question: str) -> str:
    """调用千问回答一道初中物理问题。"""
    if not question or not question.strip():
        raise ValueError("问题不能为空，请提供一道初中物理问题。")

    api_key, base_url, model = load_qwen_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUNIOR_PHYSICS_SYSTEM_PROMPT},
                {"role": "user", "content": question.strip()},
            ],
        )
        answer = response.choices[0].message.content
        return answer if answer else "千问返回了空回答。"
    except AuthenticationError:
        return "认证失败：请检查 DASHSCOPE_API_KEY 是否正确。"
    except APIConnectionError:
        return "网络连接失败：请检查网络或 QWEN_BASE_URL。"
    except RateLimitError:
        return "请求过于频繁：请稍后再试。"
    except APIError:
        return "千问 API 请求失败，请稍后再试。"
    except Exception:
        return "调用千问时发生未知错误，请稍后再试。"
