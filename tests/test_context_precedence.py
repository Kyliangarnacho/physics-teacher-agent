"""Current-query priority and historical-context precedence tests."""

from __future__ import annotations

import unittest

from src.agent import run_teacher_agent
from src.context.adapters import build_final_context_view
from src.model_client import build_conversation_messages, build_messages
from src.prompts import QUESTION_ANALYZER_SYSTEM_PROMPT
from src.tool_client import TOOL_SELECTION_INSTRUCTION
from tests.test_agent import analysis_json
from tests.test_context_agent_integration import (
    BundleAnalyzer,
    BundleAnswer,
    BundleToolAnswer,
    make_bundle,
)


class CurrentQueryPrecedenceTests(unittest.TestCase):
    def test_explicit_explanation_request_does_not_route_to_tool(self) -> None:
        bundle = make_bundle(
            summary=(
                "旧任务A：求平均速度。旧任务B：求机械效率。"
                "旧任务C：计算电流。"
            )
        )
        analyzer = BundleAnalyzer(
            calculation_required=False,
            context_relation="follow_up",
        )
        answer = BundleAnswer()
        tool_answer = BundleToolAnswer()

        result = run_teacher_agent(
            "这次不计算，只解释为什么电阻会阻碍电流。",
            analyzer_func=analyzer,
            answer_func=answer,
            tool_answer_func=tool_answer,
            context_bundle=bundle,
        )

        self.assertFalse(result["route"]["use_tools"])
        self.assertEqual(tool_answer.calls, [])
        self.assertEqual(len(answer.calls), 1)
        self.assertEqual(
            answer.calls[0]["question"],
            "这次不计算，只解释为什么电阻会阻碍电流。",
        )

    def test_last_user_is_the_only_current_task_when_history_has_a_b_c(self) -> None:
        bundle = make_bundle(
            summary="旧问题 A：求速度。旧问题 B：求效率。旧问题 C：解释电流。"
        )
        question = "现在只回答 C：为什么电阻增大会使电流减小？"

        messages = build_messages(question, context_bundle=bundle)

        self.assertEqual(messages[-1], {"role": "user", "content": question})
        self.assertEqual(
            sum(
                question in str(message.get("content", ""))
                for message in messages
            ),
            1,
        )
        self.assertIn("最后一个 user 消息是本轮唯一", messages[0]["content"])
        self.assertIn("旧问题、旧计算请求和旧工具需求", messages[0]["content"])

    def test_retrieved_old_task_is_explicitly_non_current_and_non_authoritative(self) -> None:
        view = build_final_context_view(make_bundle())
        supplemental = view["context_bundle_context"]

        self.assertIn("Old assistant content may be wrong", supplemental)
        self.assertIn("not tasks for the current turn", supplemental)

    def test_short_numbered_follow_up_can_use_previous_state(self) -> None:
        bundle = make_bundle()
        analyzer = BundleAnalyzer(context_relation="follow_up")
        analyzer.result = analysis_json(
            context_relation="follow_up",
            needs_previous_image_context=True,
            calculation_required=False,
        )
        answer = BundleAnswer()

        run_teacher_agent(
            "那②呢？",
            analyzer_func=analyzer,
            answer_func=answer,
            previous_problem_text="图示电路包含①②③三个小问。",
            previous_image_context="图中标出了②对应的开关状态。",
            context_bundle=bundle,
        )

        active_question = answer.calls[0]["question"]
        self.assertIn("当前指令：\n那②呢？", active_question)
        self.assertIn("上一活动题目", active_question)
        self.assertIn("②对应的开关状态", active_question)

    def test_new_problem_remains_current_despite_old_context(self) -> None:
        bundle = make_bundle(summary="旧题一直在计算 R1 电流。")
        analyzer = BundleAnalyzer(context_relation="new_problem")
        answer = BundleAnswer()
        question = "新题：请解释声音为什么不能在真空中传播。"

        run_teacher_agent(
            question,
            analyzer_func=analyzer,
            answer_func=answer,
            context_bundle=bundle,
        )

        self.assertEqual(answer.calls[0]["question"], question)

    def test_analyzer_prompt_states_typo_ambiguity_and_diagnosis_rules(self) -> None:
        messages = build_conversation_messages(
            QUESTION_ANALYZER_SYSTEM_PROMPT,
            "我把 I=U/R 写成了 I=R/U，这里是不是写反了？",
            context_bundle=make_bundle(),
        )
        system_prompt = messages[0]["content"]

        self.assertIn("明显且唯一的笔误或口误", system_prompt)
        self.assertIn("多个合理解释或会实质改变题意", system_prompt)
        self.assertIn("必须询问确认", system_prompt)
        self.assertIn("错误公式、概念或步骤属于教学诊断信息", system_prompt)
        self.assertEqual(
            messages[-1]["content"],
            "我把 I=U/R 写成了 I=R/U，这里是不是写反了？",
        )

    def test_tool_policy_is_strictly_current_request_scoped(self) -> None:
        policy = TOOL_SELECTION_INSTRUCTION

        self.assertIn("不计算、只解释", policy)
        self.assertIn("当前只问其中一项", policy)
        self.assertIn("不要猜测参数或调用工具", policy)
        self.assertIn("不能仅因来自当前 user 就当作正确先验", policy)

    def test_priority_rules_do_not_change_model_request_count(self) -> None:
        result = run_teacher_agent(
            "只解释欧姆定律的含义。",
            analyzer_func=BundleAnalyzer(calculation_required=False),
            answer_func=BundleAnswer(),
            context_bundle=make_bundle(),
        )

        self.assertEqual(result["trace"]["total_model_requests"], 2)


if __name__ == "__main__":
    unittest.main()
