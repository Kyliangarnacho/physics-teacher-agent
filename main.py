"""初中物理教师 Agent 第一阶段入口。"""

import os

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError


SYSTEM_MESSAGE = "你是一名初中物理教师，请使用初中物理范围内的知识，正确、清晰、简洁地回答。"
TEST_QUESTION = "一辆小车在 10 秒内行驶了 50 米，求平均速度，并简要说明计算过程。"


def load_config() -> tuple[str, str, str] | None:
    """加载并检查千问 API 配置。"""
    load_dotenv()

    config = {
        "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY"),
        "QWEN_BASE_URL": os.getenv("QWEN_BASE_URL"),
        "QWEN_MODEL": os.getenv("QWEN_MODEL"),
    }
    missing = [name for name, value in config.items() if not value]
    if missing:
        print(f"配置缺失：请在 .env 中设置 {', '.join(missing)}。")
        return None

    return (
        config["DASHSCOPE_API_KEY"],
        config["QWEN_BASE_URL"],
        config["QWEN_MODEL"],
    )


def main() -> None:
    """调用千问回答初中物理测试题。"""
    config = load_config()
    if config is None:
        return

    api_key, base_url, model = config
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": TEST_QUESTION},
            ],
        )
        answer = response.choices[0].message.content
        if answer:
            print(answer)
        else:
            print("千问返回了空回答。")
    except AuthenticationError:
        print("认证失败：请检查 DASHSCOPE_API_KEY 是否正确。")
    except APIConnectionError:
        print("网络连接失败：请检查网络或 QWEN_BASE_URL。")
    except RateLimitError:
        print("请求过于频繁：请稍后再试。")
    except APIError:
        print("千问 API 请求失败，请稍后再试。")
    except Exception:
        print("调用千问时发生未知错误，请稍后再试。")


if __name__ == "__main__":
    main()
