"""Stage 10 Conversation Service 单轮对话编排单元测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.conversation import ConversationServiceError, run_conversation_turn
from src.storage import (
    RepositoryError,
    create_conversation,
    get_agent_runs,
    get_conversation_state,
    initialize_database,
    list_messages,
)
from src.storage import repositories


def fake_agent_result(
    answer: str = "教师回答",
    status: str = "completed",
    teaching_mode: str = "solve",
    use_rag: bool = False,
    use_tools: bool = False,
    total_model_requests: int = 2,
) -> dict:
    return {
        "answer": answer,
        "analysis": {"teaching_mode": teaching_mode},
        "route": {
            "teaching_mode": teaching_mode,
            "use_rag": use_rag,
            "use_tools": use_tools,
            "should_answer": True,
        },
        "sources": [],
        "analysis_fallback": False,
        "tool_records": [],
        "tool_model_requests": 0,
        "trace": {
            "status": status,
            "total_model_requests": total_model_requests,
            "total_duration_ms": 10.0,
            "rag_searches": 0,
            "tool_executions": 0,
            "run_id": "run-fixed",
            "steps": [],
        },
    }


class FakeAgent:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = list(results) if results else [fake_agent_result()]
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, question: str, **kwargs) -> dict:
        self.calls.append({"question": question, **kwargs})
        if self.error is not None:
            raise self.error
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


class _TrackingConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.alive = True

    def execute(self, sql, params=None):
        return self._conn.execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self.alive = False
        self._conn.close()


class _TrackingFactory:
    def __init__(self, real) -> None:
        self.real = real
        self.connections: list[_TrackingConnection] = []

    def __call__(self, path=None):
        wrapped = _TrackingConnection(self.real(path))
        self.connections.append(wrapped)
        return wrapped

    def live_count(self) -> int:
        return sum(1 for connection in self.connections if connection.alive)


class ConversationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "physics_teacher.db")
        initialize_database(self.db_path)
        self.conversation = create_conversation("服务测试", path=self.db_path)
        self.conversation_id = self.conversation["id"]

    def test_first_turn_saves_full_cycle(self) -> None:
        result = run_conversation_turn(
            self.conversation_id,
            "解释欧姆定律",
            "解释欧姆定律",
            db_path=self.db_path,
            agent_func=FakeAgent(),
        )

        for key in (
            "agent_result",
            "conversation_id",
            "user_message_id",
            "assistant_message_id",
            "resolved_state",
            "history_turn_count",
            "state_context_used",
            "memory_count",
            "memory_context_used",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["conversation_id"], self.conversation_id)
        self.assertEqual(result["history_turn_count"], 0)

        messages = list_messages(self.conversation_id, path=self.db_path)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["id"], result["assistant_message_id"])
        runs = get_agent_runs(self.conversation_id, path=self.db_path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "completed")
        state = get_conversation_state(self.conversation_id, path=self.db_path)
        self.assertIsNotNone(state)
        self.assertEqual(state["hint_step"], 0)
        self.assertEqual(state["teaching_mode"], "solve")

    def test_second_turn_reads_first_turn_history(self) -> None:
        agent = FakeAgent(
            results=[
                fake_agent_result(answer="第一答"),
                fake_agent_result(answer="第二答"),
            ]
        )
        run_conversation_turn(
            self.conversation_id,
            "第一题",
            "第一题",
            db_path=self.db_path,
            agent_func=agent,
        )

        result = run_conversation_turn(
            self.conversation_id,
            "第二题",
            "第二题",
            db_path=self.db_path,
            agent_func=agent,
        )

        self.assertEqual(result["history_turn_count"], 1)
        self.assertEqual(
            agent.calls[1]["conversation_history"],
            [
                {"role": "user", "content": "第一题"},
                {"role": "assistant", "content": "第一答"},
            ],
        )

    def test_follow_up_inherits_problem_and_hint_step(self) -> None:
        agent = FakeAgent(
            results=[
                fake_agent_result(answer="第一步：写出 I=U/R。", teaching_mode="hint"),
                fake_agent_result(answer="第二步：代入 U=12V、R=6Ω。", teaching_mode="hint"),
            ]
        )
        run_conversation_turn(
            self.conversation_id,
            "12 V、6 Ω，求电流，只给第一步",
            "12 V、6 Ω，求电流，只给第一步",
            mode_override="hint",
            db_path=self.db_path,
            agent_func=agent,
        )
        self.assertEqual(
            get_conversation_state(self.conversation_id, path=self.db_path)["hint_step"],
            1,
        )

        result = run_conversation_turn(
            self.conversation_id,
            "继续下一步",
            "继续下一步",
            db_path=self.db_path,
            agent_func=agent,
        )

        self.assertTrue(result["resolved_state"].is_follow_up)
        self.assertEqual(
            result["resolved_state"].active_problem_text,
            "12 V、6 Ω，求电流，只给第一步",
        )
        self.assertEqual(result["resolved_state"].hint_step, 1)
        state = get_conversation_state(self.conversation_id, path=self.db_path)
        self.assertEqual(state["hint_step"], 2)
        self.assertEqual(state["teaching_mode"], "hint")
        self.assertEqual(state["active_problem_text"], "12 V、6 Ω，求电流，只给第一步")

    def test_new_question_replaces_state_and_clears_old_image(self) -> None:
        agent = FakeAgent()
        run_conversation_turn(
            self.conversation_id,
            "题目 A",
            "题目 A",
            confirmed_image_context="旧图上下文",
            db_path=self.db_path,
            agent_func=agent,
        )
        self.assertEqual(
            get_conversation_state(self.conversation_id, path=self.db_path)[
                "active_image_context"
            ],
            "旧图上下文",
        )

        result = run_conversation_turn(
            self.conversation_id,
            "完全无关的新题",
            "完全无关的新题",
            db_path=self.db_path,
            agent_func=agent,
        )

        self.assertEqual(result["resolved_state"].active_problem_text, "完全无关的新题")
        self.assertIsNone(result["resolved_state"].active_image_context)
        state = get_conversation_state(self.conversation_id, path=self.db_path)
        self.assertEqual(state["active_problem_text"], "完全无关的新题")
        self.assertIsNone(state["active_image_context"])

    def test_blocked_does_not_advance_state(self) -> None:
        run_conversation_turn(
            self.conversation_id,
            "12 V、6 Ω，求电流",
            "12 V、6 Ω，求电流",
            mode_override="hint",
            db_path=self.db_path,
            agent_func=FakeAgent(
                results=[fake_agent_result(answer="第一步提示", teaching_mode="hint")]
            ),
        )
        self.assertEqual(
            get_conversation_state(self.conversation_id, path=self.db_path)["hint_step"],
            1,
        )

        result = run_conversation_turn(
            self.conversation_id,
            "图中电压表读数是多少？",
            "图中电压表读数是多少？",
            db_path=self.db_path,
            agent_func=FakeAgent(
                results=[
                    fake_agent_result(
                        answer="请补充题图，或完整描述图中的物体、连接关系和已知信息。",
                        status="blocked",
                    )
                ]
            ),
        )

        state = get_conversation_state(self.conversation_id, path=self.db_path)
        self.assertEqual(state["hint_step"], 1)
        self.assertEqual(state["teaching_mode"], "hint")
        self.assertEqual(result["agent_result"]["trace"]["status"], "blocked")
        runs = get_agent_runs(self.conversation_id, path=self.db_path)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[1]["status"], "blocked")

    def test_agent_exception_keeps_user_and_records_failed_run(self) -> None:
        agent = FakeAgent(error=RuntimeError("boom"))

        with self.assertRaises(ConversationServiceError):
            run_conversation_turn(
                self.conversation_id,
                "题目",
                "题目",
                db_path=self.db_path,
                agent_func=agent,
            )

        messages = list_messages(self.conversation_id, path=self.db_path)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        runs = get_agent_runs(self.conversation_id, path=self.db_path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "failed")
        self.assertIsNone(get_conversation_state(self.conversation_id, path=self.db_path))

    def test_finalize_failure_rolls_back_but_keeps_user(self) -> None:
        def failing_finalize(*args, **kwargs):
            raise RepositoryError("会话轮次保存失败：数据库错误。")

        with mock.patch(
            "src.conversation.service.finalize_conversation_turn",
            failing_finalize,
        ):
            with self.assertRaises(ConversationServiceError):
                run_conversation_turn(
                    self.conversation_id,
                    "题目",
                    "题目",
                    db_path=self.db_path,
                    agent_func=FakeAgent(),
                )

        messages = list_messages(self.conversation_id, path=self.db_path)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(
            get_agent_runs(self.conversation_id, path=self.db_path),
            [],
        )
        self.assertIsNone(
            get_conversation_state(self.conversation_id, path=self.db_path)
        )

    def test_no_db_connection_open_during_agent_call(self) -> None:
        factory = _TrackingFactory(repositories.connect_database)
        observed: dict[str, int] = {}

        def spying_agent(question: str, **kwargs) -> dict:
            observed["live_during_agent"] = factory.live_count()
            return fake_agent_result()

        with mock.patch(
            "src.storage.repositories.connect_database",
            factory,
        ):
            run_conversation_turn(
                self.conversation_id,
                "题目",
                "题目",
                db_path=self.db_path,
                agent_func=spying_agent,
            )

        self.assertEqual(observed["live_during_agent"], 0)
        self.assertEqual(factory.live_count(), 0)

    def test_display_and_model_content_division(self) -> None:
        run_conversation_turn(
            self.conversation_id,
            "显示版题目（带格式）",
            "模型版题目",
            db_path=self.db_path,
            agent_func=FakeAgent(),
        )

        messages = list_messages(self.conversation_id, path=self.db_path)
        self.assertEqual(messages[0]["display_content"], "显示版题目（带格式）")
        self.assertEqual(messages[0]["model_content"], "模型版题目")
        self.assertEqual(messages[1]["display_content"], "教师回答")
        self.assertEqual(messages[1]["model_content"], "教师回答")

    def test_missing_conversation_no_writes(self) -> None:
        with self.assertRaises(ConversationServiceError):
            run_conversation_turn(
                "missing-conv",
                "题目",
                "题目",
                db_path=self.db_path,
                agent_func=FakeAgent(),
            )

        self.assertEqual(list_messages("missing-conv", path=self.db_path), [])
        self.assertEqual(get_agent_runs("missing-conv", path=self.db_path), [])
        self.assertIsNone(get_conversation_state("missing-conv", path=self.db_path))

    def test_errors_do_not_leak_sql_path_or_keys(self) -> None:
        with self.assertRaises(ConversationServiceError) as ctx:
            run_conversation_turn(
                "missing-conv",
                "题目",
                "题目",
                db_path=self.db_path,
            )
        message = str(ctx.exception)
        self.assertNotIn("INSERT", message)
        self.assertNotIn(self.db_path, message)
        self.assertNotIn("DASHSCOPE", message)

        agent = FakeAgent(error=RuntimeError("boom"))
        with self.assertRaises(ConversationServiceError) as ctx:
            run_conversation_turn(
                self.conversation_id,
                "题目",
                "题目",
                db_path=self.db_path,
                agent_func=agent,
            )
        message = str(ctx.exception)
        self.assertNotIn("boom", message)
        self.assertNotIn(self.db_path, message)

    def test_effective_question_contains_original_problem(self) -> None:
        agent = FakeAgent(
            results=[
                fake_agent_result(answer="第一步提示", teaching_mode="hint"),
                fake_agent_result(answer="第二步提示", teaching_mode="hint"),
            ]
        )
        run_conversation_turn(
            self.conversation_id,
            "12 V、6 Ω，求电流，只给第一步",
            "12 V、6 Ω，求电流，只给第一步",
            mode_override="hint",
            db_path=self.db_path,
            agent_func=agent,
        )
        run_conversation_turn(
            self.conversation_id,
            "继续下一步",
            "继续下一步",
            db_path=self.db_path,
            agent_func=agent,
        )

        question = agent.calls[1]["question"]
        self.assertIn("继续下一步", question)
        self.assertIn("原题", question)
        self.assertIn("12 V、6 Ω，求电流，只给第一步", question)


if __name__ == "__main__":
    unittest.main()
