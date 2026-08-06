"""Stage 10 历史与教学状态上下文链路接入单元测试。"""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from unittest import mock

from src.agent import run_teacher_agent
from src.analyzer import analyze_question_with_trace
from src.model_client import (
    build_conversation_messages,
    build_messages,
    validate_conversation_history,
)
from src.tool_client import answer_with_tools
from tests.test_agent import (
    RAG_CARDS,
    FakeAnalyzer,
    FakeAnswer,
    FakeRetriever,
    analysis_json,
)
from tests.test_tool_client import completion_response, tool_call


HISTORY = [
    {"role": "user", "content": "一辆小车 10 秒行驶 50 米，求平均速度。"},
    {"role": "assistant", "content": "平均速度 = 路程 ÷ 时间 = 50 ÷ 10 = 5 m/s。"},
]
STATE = (
    "当前活动题目：电压 12 V，电阻 6 Ω，求电流。\n"
    "当前教学模式：hint\n"
    "已完成提示步数：1"
)


class ContextAnalyzer:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        question: str,
        *,
        teaching_state_context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "teaching_state_context": teaching_state_context,
                "conversation_history": conversation_history,
            }
        )
        return self.result


class ContextAnswer:
    def __init__(self, answer: str = "承接回答") -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        question: str,
        context: str | None = None,
        mode_instruction: str | None = None,
        *,
        teaching_state_context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "context": context,
                "mode_instruction": mode_instruction,
                "teaching_state_context": teaching_state_context,
                "conversation_history": conversation_history,
            }
        )
        return self.answer


class ContextToolAnswer:
    def __init__(self, answer: str = "工具承接回答") -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        question: str,
        context: str | None = None,
        mode_instruction: str | None = None,
        *,
        teaching_state_context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "question": question,
                "context": context,
                "mode_instruction": mode_instruction,
                "teaching_state_context": teaching_state_context,
                "conversation_history": conversation_history,
            }
        )
        return {
            "answer": self.answer,
            "tool_records": [
                {
                    "tool_call_id": "call-1",
                    "name": "calculate_ohms_law",
                    "status": "success",
                }
            ],
            "model_requests": 2,
        }


def count_occurrences(messages: list[dict[str, str]], text: str) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.get("content"), str)
        and text in message["content"]
    )


class MessageBuilderContextTests(unittest.TestCase):
    def test_state_history_and_question_order(self) -> None:
        messages = build_conversation_messages(
            "基础系统提示",
            "当前问题",
            context="RAG 参考资料",
            mode_instruction="模式指令",
            teaching_state_context=STATE,
            conversation_history=HISTORY,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "system", "system", "system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[0]["content"], "基础系统提示")
        self.assertEqual(messages[1]["content"], "模式指令")
        self.assertIn("RAG 参考资料", messages[2]["content"])
        self.assertIn("教学状态", messages[3]["content"])
        self.assertIn(HISTORY[0]["content"], messages[4]["content"])
        self.assertIn(HISTORY[1]["content"], messages[5]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "当前问题"})
        self.assertEqual(count_occurrences(messages, "教学状态"), 1)

    def test_empty_history_and_state_generate_no_extra_messages(self) -> None:
        messages = build_conversation_messages(
            "系统提示",
            "问题",
            teaching_state_context=" \t",
            conversation_history=[],
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "问题"},
            ],
        )

    def test_invalid_history_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "role"):
            build_messages("问题", conversation_history=[{"role": "system", "content": "x"}])
        with self.assertRaisesRegex(ValueError, "content"):
            build_messages("问题", conversation_history=[{"role": "user", "content": 123}])
        with self.assertRaisesRegex(ValueError, "额外字段"):
            build_messages(
                "问题",
                conversation_history=[
                    {"role": "user", "content": "x", "extra": 1}
                ],
            )
        with self.assertRaisesRegex(ValueError, "必须是列表"):
            build_messages("问题", conversation_history="not-a-list")

    def test_history_is_not_mutated(self) -> None:
        history = deepcopy(HISTORY)

        build_messages("问题", conversation_history=history)
        validated = validate_conversation_history(history)

        self.assertEqual(history, HISTORY)
        self.assertEqual(validated, HISTORY)
        self.assertIsNot(validated, history)
        self.assertIsNot(validated[0], history[0])


class AnalyzerContextTests(unittest.TestCase):
    def test_analyzer_fake_receives_history_and_state(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())

        analyze_question_with_trace(
            "为什么要除以时间？",
            analyze_func=analyzer,
            teaching_state_context=STATE,
            conversation_history=HISTORY,
        )

        self.assertEqual(len(analyzer.calls), 1)
        self.assertEqual(analyzer.calls[0]["teaching_state_context"], STATE)
        self.assertEqual(analyzer.calls[0]["conversation_history"], HISTORY)

    def test_analyzer_real_model_path_receives_context(self) -> None:
        with mock.patch(
            "src.analyzer._call_analyzer_model",
            return_value=analysis_json(),
        ) as mock_call:
            analysis, fallback, _ = analyze_question_with_trace(
                "继续",
                teaching_state_context=STATE,
                conversation_history=HISTORY,
            )

        self.assertFalse(fallback)
        self.assertIsNotNone(analysis)
        mock_call.assert_called_once_with(
            "继续",
            teaching_state_context=STATE,
            learning_memory_context=None,
            conversation_history=HISTORY,
        )

    def test_old_single_arg_analyzer_still_works(self) -> None:
        calls: list[str] = []

        def old_style(question: str) -> str:
            calls.append(question)
            return analysis_json()

        analyze_question_with_trace(
            "继续",
            analyze_func=old_style,
            teaching_state_context=STATE,
            conversation_history=HISTORY,
        )

        self.assertEqual(calls, ["继续"])


class AgentContextTests(unittest.TestCase):
    def test_default_behavior_without_context_is_unchanged(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())
        answer = ContextAnswer()

        result = run_teacher_agent(
            "解释欧姆定律。",
            analyzer_func=analyzer,
            answer_func=answer,
        )

        self.assertEqual(result["answer"], "承接回答")
        self.assertIsNone(analyzer.calls[0]["teaching_state_context"])
        self.assertIsNone(analyzer.calls[0]["conversation_history"])
        self.assertIsNone(answer.calls[0]["teaching_state_context"])
        self.assertIsNone(answer.calls[0]["conversation_history"])

    def test_analyzer_and_plain_answer_receive_history_and_state(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())
        answer = ContextAnswer()

        run_teacher_agent(
            "为什么要除以时间？",
            analyzer_func=analyzer,
            answer_func=answer,
            conversation_history=HISTORY,
            teaching_state_context=STATE,
        )

        self.assertEqual(analyzer.calls[0]["teaching_state_context"], STATE)
        self.assertEqual(analyzer.calls[0]["conversation_history"], HISTORY)
        self.assertEqual(answer.calls[0]["teaching_state_context"], STATE)
        self.assertEqual(answer.calls[0]["conversation_history"], HISTORY)

    def test_rag_path_receives_context_and_retrieval(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())
        retriever = FakeRetriever(RAG_CARDS)
        answer = ContextAnswer()

        run_teacher_agent(
            "欧姆定律是什么？",
            rag_policy="force",
            analyzer_func=analyzer,
            retriever=retriever,
            answer_func=answer,
            conversation_history=HISTORY,
            teaching_state_context=STATE,
        )

        self.assertEqual(len(answer.calls), 1)
        self.assertIsNotNone(answer.calls[0]["context"])
        self.assertEqual(answer.calls[0]["teaching_state_context"], STATE)
        self.assertEqual(answer.calls[0]["conversation_history"], HISTORY)
        self.assertEqual(retriever.calls, [("欧姆定律是什么？", 3)])

    def test_tool_path_receives_context(self) -> None:
        analyzer = ContextAnalyzer(
            analysis_json(teaching_mode="solve", calculation_required=True)
        )
        tool_answer = ContextToolAnswer()

        run_teacher_agent(
            "电压 12 V，电阻 6 Ω，求电流。",
            analyzer_func=analyzer,
            tool_answer_func=tool_answer,
            conversation_history=HISTORY,
            teaching_state_context=STATE,
        )

        self.assertEqual(len(tool_answer.calls), 1)
        self.assertEqual(tool_answer.calls[0]["teaching_state_context"], STATE)
        self.assertEqual(tool_answer.calls[0]["conversation_history"], HISTORY)

    def test_blocked_path_does_not_call_final_answer(self) -> None:
        analyzer = ContextAnalyzer(analysis_json(image_required=True))
        answer = ContextAnswer()

        result = run_teacher_agent(
            "图中电压表读数是多少？",
            analyzer_func=analyzer,
            answer_func=answer,
            conversation_history=HISTORY,
            teaching_state_context=STATE,
        )

        self.assertEqual(answer.calls, [])
        self.assertEqual(result["trace"]["status"], "blocked")
        self.assertIn("请补充题图", result["answer"])

    def test_invalid_history_rejected_before_any_model_call(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())

        with self.assertRaises(ValueError):
            run_teacher_agent(
                "问题",
                analyzer_func=analyzer,
                conversation_history=[{"role": "system", "content": "x"}],
            )

        self.assertEqual(analyzer.calls, [])

    def test_input_history_is_not_mutated_by_agent(self) -> None:
        history = deepcopy(HISTORY)
        analyzer = ContextAnalyzer(analysis_json())
        answer = ContextAnswer()

        run_teacher_agent(
            "继续",
            analyzer_func=analyzer,
            answer_func=answer,
            conversation_history=history,
            teaching_state_context=STATE,
        )

        self.assertEqual(history, HISTORY)

    def test_old_signature_fakes_still_work_with_context(self) -> None:
        result = run_teacher_agent(
            "继续",
            analyzer_func=FakeAnalyzer(analysis_json()),
            answer_func=FakeAnswer("旧签名回答"),
            conversation_history=HISTORY,
            teaching_state_context=STATE,
        )

        self.assertEqual(result["answer"], "旧签名回答")

    def test_trace_does_not_contain_history_or_state_content(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())
        answer = ContextAnswer()

        result = run_teacher_agent(
            "继续",
            analyzer_func=analyzer,
            answer_func=answer,
            conversation_history=HISTORY,
            teaching_state_context=STATE,
        )

        trace_text = json.dumps(result["trace"], ensure_ascii=False)
        for forbidden in (
            "一辆小车 10 秒行驶 50 米",
            "平均速度 = 路程",
            "当前活动题目：电压 12 V",
            "已完成提示步数",
        ):
            self.assertNotIn(forbidden, trace_text)


class ToolClientContextTests(unittest.TestCase):
    def _fake_completion(self, responses):
        from tests.test_tool_client import FakeCompletion

        return FakeCompletion(responses)

    def test_tool_first_request_includes_context_once(self) -> None:
        fake = self._fake_completion(
            [
                completion_response(tool_calls=[tool_call()]),
                completion_response(content="根据欧姆定律，电流为 2 A。"),
            ]
        )

        answer_with_tools(
            "电压 12 V，电阻 6 Ω，求电流。",
            context="欧姆定律参考资料。",
            mode_instruction="给出完整答案。",
            completion_func=fake,
            teaching_state_context=STATE,
            conversation_history=HISTORY,
        )

        first = fake.calls[0]["messages"]
        self.assertEqual(
            [message["role"] for message in first],
            ["system", "system", "system", "system", "user", "assistant", "user"],
        )
        self.assertEqual(first[-1], {"role": "user", "content": "电压 12 V，电阻 6 Ω，求电流。"})
        self.assertEqual(count_occurrences(first, "教学状态"), 1)
        self.assertEqual(count_occurrences(first, HISTORY[0]["content"]), 1)
        self.assertEqual(count_occurrences(first, HISTORY[1]["content"]), 1)

    def test_tool_second_request_reuses_full_messages_without_duplicates(self) -> None:
        fake = self._fake_completion(
            [
                completion_response(tool_calls=[tool_call()]),
                completion_response(content="电流为 2 A。"),
            ]
        )

        answer_with_tools(
            "电压 12 V，电阻 6 Ω，求电流。",
            completion_func=fake,
            teaching_state_context=STATE,
            conversation_history=HISTORY,
        )

        first_messages = fake.calls[0]["messages"]
        second_messages = fake.calls[1]["messages"]
        self.assertEqual(second_messages[:-2], first_messages)
        self.assertEqual(count_occurrences(second_messages, "教学状态"), 1)
        self.assertEqual(count_occurrences(second_messages, HISTORY[0]["content"]), 1)
        self.assertEqual(count_occurrences(second_messages, HISTORY[1]["content"]), 1)

    def test_tool_result_retry_does_not_duplicate_context_or_reexecute(self) -> None:
        fake = self._fake_completion(
            [
                completion_response(tool_calls=[tool_call()]),
                RuntimeError("模拟结果回传失败"),
                completion_response(content="电流为 2 A。"),
            ]
        )

        result = answer_with_tools(
            "电压 12 V，电阻 6 Ω，求电流。",
            completion_func=fake,
            teaching_state_context=STATE,
            conversation_history=HISTORY,
        )

        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(result["model_requests"], 3)
        self.assertEqual(len(result["tool_records"]), 1)
        retry_messages = fake.calls[2]["messages"]
        self.assertEqual(count_occurrences(retry_messages, "教学状态"), 1)
        self.assertEqual(count_occurrences(retry_messages, HISTORY[0]["content"]), 1)
        self.assertEqual(retry_messages[-1]["tool_call_id"], "call-ohms")
        self.assertNotIn("tools", fake.calls[2])


if __name__ == "__main__":
    unittest.main()
