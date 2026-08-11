"""Stage 10 长期记忆检索与模型注入单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from src.agent import run_teacher_agent
from src.conversation import run_conversation_turn
from src.memory import (
    build_learning_memory_context,
    normalize_memory_content,
    retrieve_relevant_memories,
)
from src.storage import (
    deactivate_memory,
    initialize_database,
    insert_or_merge_memory,
)
from src.tool_client import answer_with_tools
from tests.test_agent import (
    RAG_CARDS,
    FakeRetriever,
    analysis_json,
)
from tests.test_tool_client import FakeCompletion, completion_response, tool_call


def _add_memory(
    db_path: str,
    *,
    memory_type: str = "weakness",
    topic: str = "机械能守恒",
    content: str = "误认为物体下落时机械能一定减小",
    confirmed: int = 1,
    active: int = 1,
) -> dict:
    return insert_or_merge_memory(
        {
            "memory_type": memory_type,
            "topic": topic,
            "content": content,
            "normalized_content": normalize_memory_content(content),
            "confidence": 0.9,
            "confirmed": confirmed,
            "active": active,
        },
        path=db_path,
    )


class RetrieveMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "memory.db")
        initialize_database(self.db_path)

    def test_only_confirmed_and_active_memories_are_recalled(self) -> None:
        _add_memory(self.db_path, topic="机械能守恒")
        _add_memory(self.db_path, topic="浮力密度", confirmed=0)
        _add_memory(self.db_path, topic="电功率计算", active=0)

        memories = retrieve_relevant_memories(
            "物体下落机械能怎样变化",
            db_path=self.db_path,
        )

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["topic"], "机械能守恒")

    def test_related_misconception_hits(self) -> None:
        _add_memory(
            self.db_path,
            memory_type="misconception",
            topic="机械能守恒条件",
            content="忽略空气阻力时物体下落机械能仍守恒",
        )

        memories = retrieve_relevant_memories(
            "忽略空气阻力时，物体下落机械能怎样变化？",
            db_path=self.db_path,
        )

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["memory_type"], "misconception")

    def test_unrelated_misconception_and_weakness_not_injected(self) -> None:
        _add_memory(
            self.db_path,
            memory_type="misconception",
            topic="欧姆定律",
            content="串联电路总电阻计算漏项",
        )
        _add_memory(
            self.db_path,
            memory_type="weakness",
            topic="凸透镜成像",
            content="实像虚像判断混淆",
        )

        memories = retrieve_relevant_memories(
            "物体下落机械能怎样变化？",
            db_path=self.db_path,
        )

        self.assertEqual(memories, [])
        self.assertIsNone(
            build_learning_memory_context(
                retrieve_relevant_memories(
                    "物体下落机械能怎样变化？",
                    db_path=self.db_path,
                )
            )
        )

    def test_preference_cross_topic_but_limited_to_two(self) -> None:
        for index in range(3):
            _add_memory(
                self.db_path,
                memory_type="preference",
                topic=f"讲解偏好 {index}",
                content="先判断电路结构再列公式",
            )

        memories = retrieve_relevant_memories(
            "物体下落机械能怎样变化？",
            db_path=self.db_path,
        )

        self.assertEqual(len(memories), 2)
        self.assertTrue(
            all(item["memory_type"] == "preference" for item in memories)
        )

    def test_topic_direct_hit_ranks_first_and_max_memories_applies(self) -> None:
        _add_memory(
            self.db_path,
            memory_type="misconception",
            topic="机械能守恒",
            content="高度降低不等于机械能减小",
        )
        _add_memory(
            self.db_path,
            memory_type="misconception",
            topic="自由落体",
            content="自由落体速度越来越大",
        )
        _add_memory(
            self.db_path,
            memory_type="misconception",
            topic="动能定理",
            content="合外力做功等于动能变化",
        )

        memories = retrieve_relevant_memories(
            "机械能守恒条件下，物体下落动能怎样变化？",
            db_path=self.db_path,
            max_memories=2,
        )

        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["topic"], "机械能守恒")

    def test_active_problem_text_helps_relatedness(self) -> None:
        _add_memory(
            self.db_path,
            memory_type="misconception",
            topic="机械能守恒条件",
            content="不能因高度降低就判断机械能减小",
        )

        memories = retrieve_relevant_memories(
            "继续",
            active_problem_text="物体下落过程中机械能守恒吗？",
            db_path=self.db_path,
        )

        self.assertEqual(len(memories), 1)


class BuildMemoryContextTests(unittest.TestCase):
    def test_context_contains_only_type_topic_content(self) -> None:
        memories = [
            {
                "memory_type": "misconception",
                "topic": "机械能守恒",
                "content": "误认为下落机械能一定减小",
                "id": "hidden-id",
                "evidence_count": 7,
                "source_conversation_id": "hidden-conv",
            }
        ]

        context = build_learning_memory_context(memories)

        self.assertIsNotNone(context)
        self.assertIn("[错误观念] 机械能守恒：误认为下落机械能一定减小", context)
        self.assertNotIn("hidden-id", context)
        self.assertNotIn("evidence_count", context)
        self.assertNotIn("source_conversation_id", context)
        self.assertNotIn("hidden-conv", context)

    def test_char_budget_drops_entries(self) -> None:
        memories = [
            {
                "memory_type": "weakness",
                "topic": f"主题{index}",
                "content": "内容" * 8,
            }
            for index in range(3)
        ]

        context = build_learning_memory_context(memories, max_chars=80)

        self.assertIsNotNone(context)
        self.assertLessEqual(len(context), 80)
        self.assertIn("主题0", context)
        self.assertNotIn("主题1", context)

    def test_empty_or_invalid_memories_return_none(self) -> None:
        self.assertIsNone(build_learning_memory_context([]))
        self.assertIsNone(
            build_learning_memory_context(
                [{"memory_type": "other", "topic": "", "content": ""}]
            )
        )

    def test_input_memories_not_modified(self) -> None:
        memories = [
            {
                "memory_type": "preference",
                "topic": "多挡电路",
                "content": "先判断串并联再列公式",
            }
        ]
        before = deepcopy(memories)

        build_learning_memory_context(memories)

        self.assertEqual(memories, before)


class ContextAnalyzer:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict] = []

    def __call__(
        self,
        question: str,
        *,
        teaching_state_context=None,
        learning_memory_context=None,
        conversation_history=None,
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "teaching_state_context": teaching_state_context,
                "learning_memory_context": learning_memory_context,
                "conversation_history": conversation_history,
            }
        )
        return self.result


class ContextAnswer:
    def __init__(self, answer: str = "承接回答") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def __call__(
        self,
        question: str,
        context=None,
        mode_instruction=None,
        *,
        teaching_state_context=None,
        learning_memory_context=None,
        conversation_history=None,
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "context": context,
                "teaching_state_context": teaching_state_context,
                "learning_memory_context": learning_memory_context,
                "conversation_history": conversation_history,
            }
        )
        return self.answer


class ContextToolAnswer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self,
        question: str,
        context=None,
        mode_instruction=None,
        *,
        teaching_state_context=None,
        learning_memory_context=None,
        conversation_history=None,
    ) -> dict:
        self.calls.append(
            {
                "question": question,
                "context": context,
                "teaching_state_context": teaching_state_context,
                "learning_memory_context": learning_memory_context,
                "conversation_history": conversation_history,
            }
        )
        return {
            "answer": "工具回答",
            "tool_records": [],
            "model_requests": 2,
        }


class MemoryContextChainTests(unittest.TestCase):
    MEMORY = "以下是与当前学习相关的长期记忆参考。"

    def test_analyzer_and_plain_answer_receive_memory(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())
        answer = ContextAnswer()

        run_teacher_agent(
            "机械能怎样变化？",
            analyzer_func=analyzer,
            answer_func=answer,
            learning_memory_context=self.MEMORY,
        )

        self.assertEqual(analyzer.calls[0]["learning_memory_context"], self.MEMORY)
        self.assertEqual(answer.calls[0]["learning_memory_context"], self.MEMORY)

    def test_rag_path_receives_memory(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())
        answer = ContextAnswer()

        run_teacher_agent(
            "欧姆定律是什么？",
            rag_policy="force",
            analyzer_func=analyzer,
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=answer,
            learning_memory_context=self.MEMORY,
        )

        self.assertIsNotNone(answer.calls[0]["context"])
        self.assertEqual(answer.calls[0]["learning_memory_context"], self.MEMORY)

    def test_tool_path_receives_memory(self) -> None:
        analyzer = ContextAnalyzer(
            analysis_json(teaching_mode="solve", calculation_required=True)
        )
        tool_answer = ContextToolAnswer()

        run_teacher_agent(
            "电压 12V，电阻 6Ω，求电流。",
            analyzer_func=analyzer,
            tool_answer_func=tool_answer,
            learning_memory_context=self.MEMORY,
        )

        self.assertEqual(tool_answer.calls[0]["learning_memory_context"], self.MEMORY)

    def test_tool_second_request_does_not_duplicate_memory(self) -> None:
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[tool_call()]),
                completion_response(content="电流为 2 A。"),
            ]
        )

        answer_with_tools(
            "电压 12V，电阻 6Ω，求电流。",
            completion_func=fake,
            learning_memory_context=self.MEMORY,
        )

        first = fake.calls[0]["messages"]
        second = fake.calls[1]["messages"]
        self.assertEqual(
            sum(
                1
                for m in first
                if isinstance(m.get("content"), str)
                and self.MEMORY in m["content"]
            ),
            1,
        )
        self.assertEqual(second[:-2], first)
        self.assertEqual(
            sum(
                1
                for m in second
                if isinstance(m.get("content"), str)
                and self.MEMORY in m["content"]
            ),
            1,
        )

    def test_current_question_is_last_user_message(self) -> None:
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[tool_call()]),
                completion_response(content="电流为 2 A。"),
            ]
        )
        question = "电压 12V，电阻 6Ω，求电流。"

        answer_with_tools(
            question,
            completion_func=fake,
            learning_memory_context=self.MEMORY,
        )

        self.assertEqual(
            fake.calls[0]["messages"][-1],
            {"role": "user", "content": question},
        )

    def test_trace_does_not_leak_full_memory(self) -> None:
        analyzer = ContextAnalyzer(analysis_json())
        answer = ContextAnswer()

        result = run_teacher_agent(
            "机械能怎样变化？",
            analyzer_func=analyzer,
            answer_func=answer,
            learning_memory_context="长期记忆内容：误认为下落机械能减小",
        )

        trace_text = json.dumps(result["trace"], ensure_ascii=False)
        self.assertNotIn("误认为下落机械能减小", trace_text)


class ServiceRetrievalDegradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "physics_teacher.db")
        initialize_database(self.db_path)
        self.conversation = insert_conversation(self.db_path)

    def test_retrieval_error_degrades_to_empty_memory(self) -> None:
        agent_calls = []

        def fake_agent(question: str, **kwargs):
            agent_calls.append(kwargs)
            return {
                "answer": "回答",
                "analysis": {},
                "route": {"teaching_mode": "solve", "use_rag": False, "use_tools": False, "should_answer": True},
                "sources": [],
                "analysis_fallback": False,
                "tool_records": [],
                "tool_model_requests": 0,
                "trace": {"status": "completed", "total_model_requests": 1, "total_duration_ms": 1.0, "run_id": "r", "steps": []},
            }

        with mock.patch(
            "src.context.manager.retrieve_relevant_memories",
            side_effect=RuntimeError("检索失败"),
        ):
            result = run_conversation_turn(
                self.conversation["id"],
                "机械能怎样变化？",
                "机械能怎样变化？",
                db_path=self.db_path,
                agent_func=fake_agent,
            )

        self.assertEqual(result["memory_count"], 0)
        self.assertFalse(result["memory_context_used"])
        self.assertIsNone(agent_calls[0].get("learning_memory_context"))


def insert_conversation(db_path: str) -> dict:
    from src.storage import create_conversation

    return create_conversation("检索测试", path=db_path)


if __name__ == "__main__":
    unittest.main()
