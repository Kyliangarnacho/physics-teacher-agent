"""初中物理教师 Agent 入口。"""

from src.model_client import answer_question


TEST_QUESTION = "一辆小车在 10 秒内行驶了 50 米，求平均速度，并简要说明计算过程。"


def main() -> None:
    """调用千问回答初中物理测试题。"""
    print(answer_question(TEST_QUESTION))


if __name__ == "__main__":
    main()
