"""Stage 11.3 ContextBundle integration across Conversation and Agent paths."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agent import run_teacher_agent
from src.context.adapters import (
    build_analyzer_context_view,
    build_final_context_view,
    build_tool_context_view,
)
from src.context.schemas import (
    ContextBundle,
    ContextTurn,
    HistoryRetrievalResult,
    RetrievedHistoryTurn,
)
from src.conversation import enqueue_conversation_turn, execute_generation_job
from src.storage import (
    create_conversation,
    get_agent_runs,
    initialize_database,
    upsert_conversation_summary,
)
from src.tool_client import answer_with_tools
from tests.test_agent import RAG_CARDS, FakeRetriever, analysis_json
from tests.test_conversation_jobs import RecordingAgent, _agent_result
from tests.test_tool_client import FakeCompletion, completion_response, tool_call


def make_bundle(*, revision: int = 1, summary: str = "旧题摘要：R1=5Ω，电源6V。") -> ContextBundle:
    bridge = ContextTurn(
        user_message_id="bridge-user",
        assistant_message_id="bridge-assistant",
        user_content="离开 Recent 但尚未摘要：滑动变阻器接入 10Ω。",
        assistant_content="已记录该条件，尚未完成摘要。",
    )
    recent = ContextTurn(
        user_message_id="recent-user",
        assistant_message_id="recent-assistant",
        user_content="最近一轮：先分析电路连接。",
        assistant_content="最近回答：R1 与变阻器串联。",
    )
    retrieved = RetrievedHistoryTurn(
        user_message_id="retrieved-user",
        assistant_message_id="retrieved-assistant",
        user_content="很早以前确认 R1 的阻值是多少？",
        assistant_content="当时对话记录为 5Ω，但旧回答不作权威知识。",
        score=2.0,
    )
    second_retrieved = RetrievedHistoryTurn(
        user_message_id="retrieved-user-2",
        assistant_message_id="retrieved-assistant-2",
        user_content="更低相关的旧计算：曾求过滑轮机械效率。",
        assistant_content="旧回答计算得到 80%，与当前请求无直接关系。",
        score=1.0,
    )
    return ContextBundle(
        rolling_summary=summary,
        bridge_history=[bridge],
        recent_history=[recent],
        retrieved_history=HistoryRetrievalResult(
            turns=[retrieved, second_retrieved],
            candidate_count=2,
            query="R1 电流",
            retrieval_used=True,
        ),
        teaching_state_context="当前活动题目：串联电路求 R1 电流。",
        learning_memory_context=(
            "以下是长期记忆参考（仅作参考，不要当成新的物理结论）：\n"
            "- [讲解偏好] 方法：先列已知量。"
        ),
        estimated_chars=500,
        budget_limit=12_000,
        current_question_chars=8,
        trimmed_components=[],
        budget_exceeded=False,
        summary_revision=revision,
        bridge_turn_count=1,
        recent_turn_count=1,
        retrieved_turn_count=2,
    )


class BundleAnalyzer:
    def __init__(
        self,
        *,
        calculation_required: bool = False,
        context_relation: str = "new_problem",
    ) -> None:
        self.calls: list[dict] = []
        self.result = analysis_json(
            calculation_required=calculation_required,
            context_relation=context_relation,
        )

    def __call__(self, question: str, **kwargs) -> str:
        self.calls.append({"question": question, **kwargs})
        return self.result


class BundleAnswer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, question: str, **kwargs) -> str:
        self.calls.append({"question": question, **kwargs})
        return "使用固定上下文回答"


class BundleToolAnswer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, question: str, **kwargs) -> dict:
        self.calls.append({"question": question, **kwargs})
        return {
            "answer": "工具路径回答",
            "tool_records": [],
            "model_requests": 2,
        }


class AgentBundleIntegrationTests(unittest.TestCase):
    def test_projection_sizes_and_component_boundaries(self) -> None:
        bundle = make_bundle(summary="关键旧条件。" * 200)

        analyzer_view = build_analyzer_context_view(bundle)
        tool_view = build_tool_context_view(
            bundle,
            {"context_relation": "follow_up"},
        )
        final_view = build_final_context_view(bundle)

        self.assertLess(tool_view["context_chars"], analyzer_view["context_chars"])
        self.assertLess(tool_view["context_chars"], final_view["context_chars"])
        self.assertIsNone(tool_view["learning_memory_context"])
        self.assertIn("很早以前确认 R1", tool_view["context_bundle_context"])
        self.assertNotIn("更低相关的旧计算", tool_view["context_bundle_context"])
        self.assertIn("更低相关的旧计算", final_view["context_bundle_context"])
        self.assertIs(analyzer_view["context_bundle"], bundle)
        self.assertIs(tool_view["context_bundle"], bundle)
        self.assertIs(final_view["context_bundle"], bundle)

    def test_analyzer_and_plain_final_consume_same_bundle(self) -> None:
        bundle = make_bundle()
        analyzer = BundleAnalyzer()
        answer = BundleAnswer()

        result = run_teacher_agent(
            "现在通过 R1 的电流是多少？",
            analyzer_func=analyzer,
            answer_func=answer,
            context_bundle=bundle,
        )

        self.assertEqual(result["answer"], "使用固定上下文回答")
        for call in (analyzer.calls[0], answer.calls[0]):
            self.assertIs(call["context_bundle"], bundle)
            self.assertIn("[Conversation Summary]", call["context_bundle_context"])
            self.assertIn("[Unsummarized Recent Overflow / Bridge History]", call["context_bundle_context"])
            self.assertIn("[Retrieved Conversation History]", call["context_bundle_context"])
            self.assertIn("更低相关的旧计算", call["context_bundle_context"])
            self.assertNotIn("最近一轮：先分析电路连接", call["context_bundle_context"])
            self.assertEqual(
                call["conversation_history"],
                [
                    {"role": "user", "content": "最近一轮：先分析电路连接。"},
                    {"role": "assistant", "content": "最近回答：R1 与变阻器串联。"},
                ],
            )
            self.assertIn("当前活动题目", call["teaching_state_context"])
            self.assertIn("讲解偏好", call["learning_memory_context"])

    def test_follow_up_tool_path_receives_narrow_view_from_identical_bundle(self) -> None:
        bundle = make_bundle()
        analyzer = BundleAnalyzer(
            calculation_required=True,
            context_relation="follow_up",
        )
        tool_answer = BundleToolAnswer()

        result = run_teacher_agent(
            "现在通过 R1 的电流是多少？",
            analyzer_func=analyzer,
            tool_answer_func=tool_answer,
            context_bundle=bundle,
        )

        self.assertEqual(result["answer"], "工具路径回答")
        self.assertIs(analyzer.calls[0]["context_bundle"], bundle)
        self.assertIs(tool_answer.calls[0]["context_bundle"], bundle)
        self.assertIsNone(tool_answer.calls[0]["learning_memory_context"])
        self.assertIn("当前活动题目", tool_answer.calls[0]["teaching_state_context"])
        self.assertIn("很早以前确认 R1", tool_answer.calls[0]["context_bundle_context"])
        self.assertNotIn("更低相关的旧计算", tool_answer.calls[0]["context_bundle_context"])

    def test_new_problem_tool_view_excludes_old_problem_context(self) -> None:
        bundle = make_bundle()
        analyzer = BundleAnalyzer(calculation_required=True)
        tool_answer = BundleToolAnswer()

        run_teacher_agent(
            "一辆新小车 10 秒行驶 50 米，求速度。",
            analyzer_func=analyzer,
            tool_answer_func=tool_answer,
            context_bundle=bundle,
        )

        call = tool_answer.calls[0]
        self.assertIs(call["context_bundle"], bundle)
        self.assertIsNone(call["context_bundle_context"])
        self.assertIsNone(call["teaching_state_context"])
        self.assertIsNone(call["learning_memory_context"])
        self.assertEqual(call["conversation_history"], [])

    def test_rag_final_keeps_history_separate_from_physics_sources(self) -> None:
        bundle = make_bundle()
        analyzer = BundleAnalyzer()
        answer = BundleAnswer()

        result = run_teacher_agent(
            "现在通过 R1 的电流是多少？",
            rag_policy="force",
            analyzer_func=analyzer,
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=answer,
            context_bundle=bundle,
        )

        self.assertIs(answer.calls[0]["context_bundle"], bundle)
        self.assertIn("欧姆定律", answer.calls[0]["context"])
        self.assertNotIn("很早以前确认", answer.calls[0]["context"])
        self.assertNotIn("很早以前确认", json.dumps(result["sources"], ensure_ascii=False))
        self.assertEqual(result["trace"]["total_model_requests"], 2)

    def test_trace_records_safe_wide_context_metadata(self) -> None:
        bundle = make_bundle().model_copy(
            update={
                "rolling_summary": None,
                "summary_revision": None,
                "budget_exceeded": True,
                "trimmed_components": ["rolling_summary"],
            }
        )
        analyzer = BundleAnalyzer()
        answer = BundleAnswer()

        result = run_teacher_agent(
            "现在只解释 R1 的作用。",
            analyzer_func=analyzer,
            answer_func=answer,
            context_bundle=bundle,
        )

        metadata = result["trace"]["context_metadata"]
        self.assertEqual(
            metadata,
            {
                "summary_revision": None,
                "summary_present": False,
                "bridge_turn_count": 1,
                "recent_turn_count": 1,
                "retrieved_turn_count": 2,
                "history_retrieval_used": True,
                "context_estimated_chars": 500,
                "budget_limit": 12_000,
                "budget_exceeded": True,
                "trimmed_components": ["rolling_summary"],
                "analyzer_context_chars": build_analyzer_context_view(bundle)[
                    "context_chars"
                ],
                "tool_context_chars": None,
                "final_context_chars": build_final_context_view(bundle)[
                    "context_chars"
                ],
            },
        )
        serialized = json.dumps(result["trace"], ensure_ascii=False)
        for unsafe_content in (
            "很早以前确认 R1",
            "最近一轮：先分析电路连接",
            "讲解偏好",
        ):
            self.assertNotIn(unsafe_content, serialized)
        self.assertEqual(result["trace"]["total_model_requests"], 2)

    def test_trace_records_narrow_tool_projection_only_for_tool_path(self) -> None:
        bundle = make_bundle()
        analyzer = BundleAnalyzer(
            calculation_required=True,
            context_relation="follow_up",
        )
        tool_answer = BundleToolAnswer()

        result = run_teacher_agent(
            "那现在求 R1 电流。",
            analyzer_func=analyzer,
            tool_answer_func=tool_answer,
            context_bundle=bundle,
        )

        metadata = result["trace"]["context_metadata"]
        self.assertEqual(
            metadata["analyzer_context_chars"],
            build_analyzer_context_view(bundle)["context_chars"],
        )
        self.assertEqual(
            metadata["tool_context_chars"],
            build_tool_context_view(
                bundle,
                {"context_relation": "follow_up"},
            )["context_chars"],
        )
        self.assertIsNone(metadata["final_context_chars"])
        self.assertEqual(result["trace"]["total_model_requests"], 3)

    def test_tool_selection_and_result_reuse_one_message_snapshot(self) -> None:
        bundle = make_bundle()
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[tool_call()]),
                completion_response(content="工具结果回答"),
            ]
        )

        result = answer_with_tools(
            "现在通过 R1 的电流是多少？",
            completion_func=fake,
            context_bundle=bundle,
        )

        self.assertEqual(result["model_requests"], 2)
        first = fake.calls[0]["messages"]
        second = fake.calls[1]["messages"]
        self.assertEqual(second[:-2], first)
        for messages in (first, second):
            serialized = json.dumps(messages, ensure_ascii=False)
            self.assertEqual(serialized.count("旧题摘要：R1=5Ω，电源6V。"), 1)
            self.assertEqual(serialized.count("最近一轮：先分析电路连接。"), 1)
            self.assertNotIn("讲解偏好", serialized)
            self.assertNotIn("更低相关的旧计算", serialized)
            user_messages = [item for item in messages if item["role"] == "user"]
            self.assertEqual(user_messages[-1]["content"], "现在通过 R1 的电流是多少？")
            self.assertEqual(serialized.count("现在通过 R1 的电流是多少？"), 1)
        first_serialized = json.dumps(first, ensure_ascii=False)
        self.assertIn("旧问题、旧工具需求不得视为本轮待执行任务", first_serialized)
        self.assertIn("只为当前用户请求选择必要工具", first_serialized)

    def test_old_calculation_tasks_do_not_become_extra_tool_calls(self) -> None:
        bundle = make_bundle(
            summary=(
                "已完成旧任务一：求平均速度。"
                "已完成旧任务二：求机械效率。"
                "已完成旧任务三：求焦距。"
            )
        )
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[tool_call()]),
                completion_response(content="只回答当前欧姆定律问题"),
            ]
        )

        result = answer_with_tools(
            "当前只求 12V、6Ω 电路中的电流。",
            completion_func=fake,
            context_bundle=bundle,
        )

        self.assertEqual(len(result["tool_records"]), 1)
        self.assertEqual(result["model_requests"], 2)
        prompt = json.dumps(fake.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("已经完成的计算", prompt)
        self.assertIn("只为当前用户请求选择必要工具", prompt)


class ConversationBundleIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "context-agent.db")
        initialize_database(self.db_path)
        self.conversation = create_conversation("Context 接入", path=self.db_path)

    def _enqueue(self, question: str):
        return enqueue_conversation_turn(
            self.conversation["id"],
            question,
            question,
            db_path=self.db_path,
        )

    def test_generation_execution_builds_bundle_exactly_once(self) -> None:
        enqueued = self._enqueue("现在通过 R1 的电流是多少？")
        bundle = make_bundle()
        agent = RecordingAgent()

        with mock.patch(
            "src.conversation.service.build_context_bundle",
            return_value=bundle,
        ) as build:
            execute_generation_job(
                enqueued["generation_job_id"],
                db_path=self.db_path,
                agent_func=agent,
            )

        build.assert_called_once_with(
            self.conversation["id"],
            "现在通过 R1 的电流是多少？",
            previous_state=None,
            path=self.db_path,
        )
        self.assertIs(agent.calls[0]["context_bundle"], bundle)
        self.assertNotIn("现在通过 R1", str(agent.calls[0]["conversation_history"]))

    def test_summary_change_does_not_mutate_snapshot_and_next_turn_rebuilds(self) -> None:
        upsert_conversation_summary(
            self.conversation["id"],
            "摘要版本一",
            path=self.db_path,
        )
        first_seen: list[ContextBundle] = []

        def first_agent(question: str, **kwargs):
            snapshot = kwargs["context_bundle"]
            first_seen.append(snapshot)
            upsert_conversation_summary(
                self.conversation["id"],
                "摘要版本二",
                path=self.db_path,
            )
            self.assertEqual(snapshot.summary_revision, 1)
            self.assertEqual(snapshot.rolling_summary, "摘要版本一")
            return _agent_result("第一答")

        first = self._enqueue("第一问")
        execute_generation_job(
            first["generation_job_id"],
            db_path=self.db_path,
            agent_func=first_agent,
        )
        self.assertEqual(first_seen[0].summary_revision, 1)

        second_agent = RecordingAgent()
        second = self._enqueue("第二问")
        execute_generation_job(
            second["generation_job_id"],
            db_path=self.db_path,
            agent_func=second_agent,
        )
        next_snapshot = second_agent.calls[0]["context_bundle"]
        self.assertEqual(next_snapshot.summary_revision, 2)
        self.assertEqual(next_snapshot.rolling_summary, "摘要版本二")

    def test_persisted_trace_metadata_is_isolated_per_conversation(self) -> None:
        conversation_b = create_conversation("Context B", path=self.db_path)
        first = self._enqueue("A 的当前问题")
        second = enqueue_conversation_turn(
            conversation_b["id"],
            "B 的当前问题",
            "B 的当前问题",
            db_path=self.db_path,
        )
        bundle_a = make_bundle(revision=1, summary="A 专属摘要")
        bundle_b = make_bundle(revision=2, summary="B 专属摘要")

        def agent(question: str, **kwargs):
            return run_teacher_agent(
                question,
                analyzer_func=BundleAnalyzer(),
                answer_func=BundleAnswer(),
                **kwargs,
            )

        with mock.patch(
            "src.conversation.service.build_context_bundle",
            side_effect=[bundle_a, bundle_b],
        ):
            execute_generation_job(
                first["generation_job_id"],
                db_path=self.db_path,
                agent_func=agent,
            )
            execute_generation_job(
                second["generation_job_id"],
                db_path=self.db_path,
                agent_func=agent,
            )

        trace_a = get_agent_runs(self.conversation["id"], path=self.db_path)[0][
            "trace_json"
        ]
        trace_b = get_agent_runs(conversation_b["id"], path=self.db_path)[0][
            "trace_json"
        ]
        self.assertEqual(trace_a["context_metadata"]["summary_revision"], 1)
        self.assertEqual(trace_b["context_metadata"]["summary_revision"], 2)
        self.assertNotIn("B 专属摘要", json.dumps(trace_a, ensure_ascii=False))
        self.assertNotIn("A 专属摘要", json.dumps(trace_b, ensure_ascii=False))

    def test_context_build_failure_marks_job_failed_without_calling_agent(self) -> None:
        enqueued = self._enqueue("问题")
        agent = RecordingAgent()

        with mock.patch(
            "src.conversation.service.build_context_bundle",
            side_effect=RuntimeError("private database detail"),
        ):
            with self.assertRaisesRegex(Exception, "上下文构建失败"):
                execute_generation_job(
                    enqueued["generation_job_id"],
                    db_path=self.db_path,
                    agent_func=agent,
                )

        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
