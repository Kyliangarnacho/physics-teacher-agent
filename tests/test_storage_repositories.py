"""Stage 10 Repository 数据访问层单元测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest import mock

from src.storage import (
    RepositoryError,
    clear_conversation_contents,
    connect_database,
    create_conversation,
    deactivate_memory,
    delete_conversation,
    delete_memory,
    finalize_conversation_turn,
    get_agent_runs,
    get_conversation,
    get_conversation_state,
    get_recent_messages,
    initialize_database,
    insert_agent_run,
    insert_message,
    insert_or_merge_memory,
    list_conversations,
    list_memories,
    list_messages,
    rename_conversation,
    reset_conversation_state,
    upsert_conversation_state,
)
from src.storage import repositories


_REAL_CONNECTION = repositories._connection


class _FlakyConnection:
    """代理 sqlite3.Connection，让指定 SQL 抛错以验证事务回滚。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql, params=None):
        if "UPDATE conversations" in str(sql):
            raise sqlite3.OperationalError("模拟更新失败")
        return self._conn.execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


@contextmanager
def _flaky_connection(path):
    with _REAL_CONNECTION(path) as conn:
        yield _FlakyConnection(conn)


class RepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "physics_teacher.db")
        initialize_database(self.db_path)

    def mark_archived(self, conversation_id: str) -> None:
        conn = connect_database(self.db_path)
        try:
            conn.execute(
                "UPDATE conversations SET archived = 1 WHERE id = ?",
                (conversation_id,),
            )
            conn.commit()
        finally:
            conn.close()


class ConversationRepositoryTests(RepositoryTestCase):
    def test_create_conversation_generates_id_and_utc_timestamps(self) -> None:
        conversation = create_conversation("力学复习", path=self.db_path)

        self.assertRegex(conversation["id"], r"^[0-9a-f]{32}$")
        self.assertEqual(conversation["title"], "力学复习")
        self.assertEqual(conversation["archived"], 0)
        for name in ("created_at", "updated_at"):
            parsed = datetime.fromisoformat(conversation[name])
            self.assertIsNotNone(parsed.utcoffset())
            self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_get_conversation_returns_none_when_missing(self) -> None:
        self.assertIsNone(get_conversation("missing-id", path=self.db_path))

    def test_list_conversations_excludes_archived_and_orders_by_updated_at_desc(
        self,
    ) -> None:
        first = create_conversation("第一", path=self.db_path)
        second = create_conversation("第二", path=self.db_path)

        self.assertEqual(
            [item["id"] for item in list_conversations(path=self.db_path)],
            [second["id"], first["id"]],
        )

        self.mark_archived(second["id"])
        self.assertEqual(
            [item["id"] for item in list_conversations(path=self.db_path)],
            [first["id"]],
        )
        self.assertEqual(
            [
                item["id"]
                for item in list_conversations(include_archived=True, path=self.db_path)
            ],
            [second["id"], first["id"]],
        )

    def test_rename_conversation_updates_title_and_updated_at(self) -> None:
        with mock.patch(
            "src.storage.repositories._utc_now_iso",
            side_effect=[
                "2026-08-06T08:00:00+00:00",
                "2026-08-06T08:00:10+00:00",
            ],
        ):
            conversation = create_conversation("旧标题", path=self.db_path)
            renamed = rename_conversation(
                conversation["id"],
                "新标题",
                path=self.db_path,
            )

        self.assertEqual(renamed["title"], "新标题")
        self.assertEqual(renamed["created_at"], "2026-08-06T08:00:00+00:00")
        self.assertEqual(renamed["updated_at"], "2026-08-06T08:00:10+00:00")

    def test_rename_missing_conversation_raises(self) -> None:
        with self.assertRaises(RepositoryError):
            rename_conversation("missing", "标题", path=self.db_path)

    def test_delete_conversation_returns_bool(self) -> None:
        conversation = create_conversation("删除测试", path=self.db_path)

        self.assertTrue(delete_conversation(conversation["id"], path=self.db_path))
        self.assertIsNone(get_conversation(conversation["id"], path=self.db_path))
        self.assertFalse(delete_conversation(conversation["id"], path=self.db_path))


class MessageRepositoryTests(RepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.conversation = create_conversation("消息测试", path=self.db_path)

    def test_insert_message_returns_record_and_updates_conversation_updated_at(
        self,
    ) -> None:
        with mock.patch(
            "src.storage.repositories._utc_now_iso",
            side_effect=[
                "2026-08-06T08:00:00+00:00",
                "2026-08-06T08:00:05+00:00",
            ],
        ):
            created = create_conversation("时间测试", path=self.db_path)
            message = insert_message(
                {
                    "conversation_id": created["id"],
                    "role": "user",
                    "display_content": "题目",
                    "model_content": "题目",
                },
                path=self.db_path,
            )

        updated = get_conversation(created["id"], path=self.db_path)
        self.assertEqual(message["created_at"], "2026-08-06T08:00:05+00:00")
        self.assertEqual(updated["created_at"], "2026-08-06T08:00:00+00:00")
        self.assertEqual(updated["updated_at"], "2026-08-06T08:00:05+00:00")

    def test_message_json_chinese_round_trip_and_stored_without_escapes(
        self,
    ) -> None:
        metadata = {
            "描述": "串联电路",
            "数量": 3,
            "备注": ["中文", "物理"],
        }
        insert_message(
            {
                "conversation_id": self.conversation["id"],
                "role": "user",
                "display_content": "看图回答",
                "model_content": "看图回答",
                "image_metadata_json": metadata,
            },
            path=self.db_path,
        )

        messages = list_messages(self.conversation["id"], path=self.db_path)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["image_metadata_json"], metadata)

        conn = connect_database(self.db_path)
        try:
            raw = conn.execute(
                "SELECT image_metadata_json FROM messages"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertIn("串联电路", raw)
        self.assertNotIn("\\u4e32", raw)

    def test_list_messages_orders_by_created_at_ascending(self) -> None:
        for index, time in enumerate(
            (
                "2026-08-06T08:00:30+00:00",
                "2026-08-06T08:00:10+00:00",
                "2026-08-06T08:00:20+00:00",
            ),
            start=1,
        ):
            insert_message(
                {
                    "id": f"msg-{index}",
                    "conversation_id": self.conversation["id"],
                    "role": "user",
                    "display_content": f"内容{index}",
                    "model_content": f"内容{index}",
                    "created_at": time,
                },
                path=self.db_path,
            )

        messages = list_messages(self.conversation["id"], path=self.db_path)
        self.assertEqual(
            [item["id"] for item in messages],
            ["msg-2", "msg-3", "msg-1"],
        )

    def test_get_recent_messages_returns_latest_n_in_chronological_order(
        self,
    ) -> None:
        for index in range(1, 6):
            insert_message(
                {
                    "id": f"msg-{index}",
                    "conversation_id": self.conversation["id"],
                    "role": "assistant",
                    "display_content": f"回答{index}",
                    "model_content": f"回答{index}",
                    "created_at": f"2026-08-06T08:00:{index:02d}+00:00",
                },
                path=self.db_path,
            )

        recent = get_recent_messages(
            self.conversation["id"],
            limit=3,
            path=self.db_path,
        )
        self.assertEqual(
            [item["id"] for item in recent],
            ["msg-3", "msg-4", "msg-5"],
        )

    def test_get_recent_messages_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(RepositoryError):
            get_recent_messages(
                self.conversation["id"],
                limit=0,
                path=self.db_path,
            )

    def test_insert_message_missing_conversation_raises_clean_error(self) -> None:
        with self.assertRaises(RepositoryError) as ctx:
            insert_message(
                {
                    "conversation_id": "missing-conv",
                    "role": "user",
                    "display_content": "内容",
                    "model_content": "内容",
                },
                path=self.db_path,
            )

        message = str(ctx.exception)
        self.assertNotIn("INSERT", message)
        self.assertNotIn(str(Path(self.db_path).parent), message)


class AgentRunRepositoryTests(RepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.conversation = create_conversation("运行测试", path=self.db_path)
        self.message = insert_message(
            {
                "conversation_id": self.conversation["id"],
                "role": "user",
                "display_content": "求电流",
                "model_content": "求电流",
            },
            path=self.db_path,
        )

    def test_insert_and_get_agent_runs_with_json_fields(self) -> None:
        run = insert_agent_run(
            {
                "conversation_id": self.conversation["id"],
                "user_message_id": self.message["id"],
                "assistant_message_id": "asm-1",
                "status": "completed",
                "teaching_mode": "solve",
                "use_rag": True,
                "use_tools": True,
                "total_model_requests": 3,
                "total_duration_ms": 1234.5,
                "sources_json": [{"id": "KB-ELEC-001", "topic": "欧姆定律"}],
                "tool_records_json": [
                    {"name": "calculate_ohms_law", "status": "success"}
                ],
                "trace_json": {
                    "run_id": "run-x",
                    "steps": [{"name": "analyzer", "状态": "成功"}],
                },
                "analysis_json": {
                    "physics_topic": "电学",
                    "question_type": "计算题",
                },
            },
            path=self.db_path,
        )

        self.assertRegex(run["run_id"], r"^[0-9a-f]{32}$")
        runs = get_agent_runs(self.conversation["id"], path=self.db_path)
        self.assertEqual(len(runs), 1)
        stored = runs[0]
        self.assertEqual(stored["status"], "completed")
        self.assertEqual(stored["teaching_mode"], "solve")
        self.assertEqual(stored["use_rag"], 1)
        self.assertEqual(stored["use_tools"], 1)
        self.assertEqual(stored["total_model_requests"], 3)
        self.assertEqual(stored["total_duration_ms"], 1234.5)
        self.assertEqual(stored["sources_json"][0]["topic"], "欧姆定律")
        self.assertEqual(
            stored["tool_records_json"][0]["name"],
            "calculate_ohms_law",
        )
        self.assertEqual(stored["trace_json"]["steps"][0]["状态"], "成功")
        self.assertEqual(stored["analysis_json"]["physics_topic"], "电学")

    def test_insert_agent_run_missing_conversation_raises(self) -> None:
        with self.assertRaises(RepositoryError):
            insert_agent_run(
                {
                    "conversation_id": "missing",
                    "user_message_id": "user-1",
                    "status": "completed",
                },
                path=self.db_path,
            )


class ConversationStateRepositoryTests(RepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.conversation = create_conversation("状态测试", path=self.db_path)

    def test_upsert_and_get_conversation_state(self) -> None:
        upsert_conversation_state(
            {
                "conversation_id": self.conversation["id"],
                "active_problem_text": "求电流",
                "active_image_context": "电路图",
                "teaching_mode": "solve",
                "hint_step": 2,
            },
            path=self.db_path,
        )

        state = get_conversation_state(self.conversation["id"], path=self.db_path)
        self.assertEqual(state["active_problem_text"], "求电流")
        self.assertEqual(state["active_image_context"], "电路图")
        self.assertEqual(state["teaching_mode"], "solve")
        self.assertEqual(state["hint_step"], 2)

        upsert_conversation_state(
            {
                "conversation_id": self.conversation["id"],
                "active_problem_text": "求电阻",
                "hint_step": 3,
            },
            path=self.db_path,
        )

        state = get_conversation_state(self.conversation["id"], path=self.db_path)
        self.assertEqual(state["active_problem_text"], "求电阻")
        self.assertEqual(state["hint_step"], 3)
        self.assertEqual(state["active_image_context"], "电路图")
        self.assertEqual(state["teaching_mode"], "solve")

    def test_upsert_state_for_missing_conversation_raises(self) -> None:
        with self.assertRaises(RepositoryError):
            upsert_conversation_state(
                {"conversation_id": "missing"},
                path=self.db_path,
            )

    def test_reset_conversation_state(self) -> None:
        upsert_conversation_state(
            {"conversation_id": self.conversation["id"]},
            path=self.db_path,
        )
        self.assertIsNotNone(
            get_conversation_state(self.conversation["id"], path=self.db_path)
        )

        self.assertTrue(
            reset_conversation_state(self.conversation["id"], path=self.db_path)
        )
        self.assertIsNone(
            get_conversation_state(self.conversation["id"], path=self.db_path)
        )
        self.assertFalse(
            reset_conversation_state(self.conversation["id"], path=self.db_path)
        )


class MemoryRepositoryTests(RepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.conversation = create_conversation("记忆测试", path=self.db_path)
        self.message = insert_message(
            {
                "conversation_id": self.conversation["id"],
                "role": "user",
                "display_content": "又错了",
                "model_content": "又错了",
            },
            path=self.db_path,
        )

    def test_first_memory_insert_uses_defaults(self) -> None:
        memory = insert_or_merge_memory(
            {
                "memory_type": "mistake",
                "topic": "欧姆定律",
                "content": "学生把公式记反",
                "normalized_content": "欧姆定律公式记错",
                "confidence": 0.8,
            },
            path=self.db_path,
        )

        self.assertRegex(memory["id"], r"^[0-9a-f]{32}$")
        self.assertEqual(memory["evidence_count"], 1)
        self.assertEqual(memory["confirmed"], 1)
        self.assertEqual(memory["active"], 1)
        self.assertEqual(len(list_memories(path=self.db_path)), 1)

    def test_merge_memory_increments_evidence_and_updates_fields_keeping_id(
        self,
    ) -> None:
        first = insert_or_merge_memory(
            {
                "memory_type": "mistake",
                "topic": "欧姆定律",
                "content": "旧内容",
                "normalized_content": "欧姆定律公式记错",
                "confidence": 0.8,
                "source_conversation_id": self.conversation["id"],
                "source_message_id": self.message["id"],
                "created_at": "2026-08-06T08:00:00+00:00",
            },
            path=self.db_path,
        )
        second = insert_or_merge_memory(
            {
                "memory_type": "mistake",
                "topic": "欧姆定律",
                "content": "新内容",
                "normalized_content": "欧姆定律公式记错",
                "confidence": 0.95,
                "source_conversation_id": None,
                "source_message_id": None,
                "confirmed": 0,
                "updated_at": "2026-08-06T08:00:30+00:00",
            },
            path=self.db_path,
        )

        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["evidence_count"], 2)
        self.assertEqual(second["content"], "新内容")
        self.assertEqual(second["confidence"], 0.95)
        self.assertEqual(second["confirmed"], 0)
        self.assertEqual(second["created_at"], "2026-08-06T08:00:00+00:00")
        self.assertEqual(second["updated_at"], "2026-08-06T08:00:30+00:00")
        self.assertEqual(len(list_memories(path=self.db_path)), 1)

    def test_memory_merge_never_creates_duplicate_rows(self) -> None:
        for index in range(3):
            insert_or_merge_memory(
                {
                    "memory_type": "mistake",
                    "topic": "欧姆定律",
                    "content": f"第 {index + 1} 次出现",
                    "normalized_content": "欧姆定律公式记错",
                    "confidence": 0.8 + index * 0.05,
                },
                path=self.db_path,
            )

        memories = list_memories(path=self.db_path)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["evidence_count"], 3)

    def test_deactivate_and_list_memory_filters(self) -> None:
        memory = insert_or_merge_memory(
            {
                "memory_type": "mistake",
                "topic": "欧姆定律",
                "content": "内容",
                "normalized_content": "欧姆定律公式记错",
                "confidence": 0.9,
            },
            path=self.db_path,
        )

        deactivated = deactivate_memory(memory["id"], path=self.db_path)
        self.assertEqual(deactivated["active"], 0)

        active_ids = [
            item["id"] for item in list_memories(path=self.db_path)
        ]
        self.assertNotIn(memory["id"], active_ids)
        all_ids = [
            item["id"]
            for item in list_memories(include_inactive=True, path=self.db_path)
        ]
        self.assertIn(memory["id"], all_ids)
        filtered = list_memories(
            memory_type="mistake",
            topic="欧姆定律",
            include_inactive=True,
            path=self.db_path,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(
            list_memories(memory_type="other", path=self.db_path),
            [],
        )

    def test_delete_memory(self) -> None:
        memory = insert_or_merge_memory(
            {
                "memory_type": "mistake",
                "topic": "欧姆定律",
                "content": "内容",
                "normalized_content": "欧姆定律公式记错",
                "confidence": 0.9,
            },
            path=self.db_path,
        )

        self.assertTrue(delete_memory(memory["id"], path=self.db_path))
        self.assertEqual(list_memories(path=self.db_path), [])
        self.assertFalse(delete_memory(memory["id"], path=self.db_path))


class CascadeAndTransactionTests(RepositoryTestCase):
    def test_delete_conversation_cascades_and_keeps_memories(self) -> None:
        conversation = create_conversation("级联测试", path=self.db_path)
        user_message = insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "user",
                "display_content": "题目",
                "model_content": "题目",
            },
            path=self.db_path,
        )
        assistant_message = insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "assistant",
                "display_content": "回答",
                "model_content": "回答",
            },
            path=self.db_path,
        )
        insert_agent_run(
            {
                "conversation_id": conversation["id"],
                "user_message_id": user_message["id"],
                "assistant_message_id": assistant_message["id"],
                "status": "completed",
                "use_rag": False,
                "use_tools": False,
                "total_model_requests": 2,
                "total_duration_ms": 100.0,
            },
            path=self.db_path,
        )
        upsert_conversation_state(
            {"conversation_id": conversation["id"], "hint_step": 1},
            path=self.db_path,
        )
        memory = insert_or_merge_memory(
            {
                "memory_type": "mistake",
                "topic": "欧姆定律",
                "content": "内容",
                "normalized_content": "欧姆定律公式记错",
                "confidence": 0.9,
                "source_conversation_id": conversation["id"],
                "source_message_id": user_message["id"],
            },
            path=self.db_path,
        )

        self.assertTrue(delete_conversation(conversation["id"], path=self.db_path))
        self.assertEqual(
            list_messages(conversation["id"], path=self.db_path),
            [],
        )
        self.assertEqual(
            get_agent_runs(conversation["id"], path=self.db_path),
            [],
        )
        self.assertIsNone(
            get_conversation_state(conversation["id"], path=self.db_path)
        )

        remaining = list_memories(path=self.db_path)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], memory["id"])
        self.assertEqual(
            remaining[0]["source_conversation_id"],
            conversation["id"],
        )

    def test_failed_write_rolls_back_transaction(self) -> None:
        conversation = create_conversation("事务测试", path=self.db_path)
        insert_message(
            {
                "id": "msg-dup",
                "conversation_id": conversation["id"],
                "role": "user",
                "display_content": "第一次",
                "model_content": "第一次",
                "created_at": "2026-08-06T08:00:00+00:00",
            },
            path=self.db_path,
        )

        with mock.patch(
            "src.storage.repositories._connection",
            _flaky_connection,
        ):
            with self.assertRaises(RepositoryError):
                insert_message(
                    {
                        "id": "msg-second",
                        "conversation_id": conversation["id"],
                        "role": "user",
                        "display_content": "第二次",
                        "model_content": "第二次",
                        "created_at": "2026-08-06T08:00:30+00:00",
                    },
                    path=self.db_path,
                )

        self.assertEqual(
            len(list_messages(conversation["id"], path=self.db_path)),
            1,
        )
        updated = get_conversation(conversation["id"], path=self.db_path)
        self.assertEqual(updated["updated_at"], "2026-08-06T08:00:00+00:00")

    def test_errors_do_not_expose_sql_or_absolute_path(self) -> None:
        with self.assertRaises(RepositoryError) as ctx:
            insert_message(
                {
                    "conversation_id": "missing",
                    "role": "user",
                    "display_content": "内容",
                    "model_content": "内容",
                },
                path=self.db_path,
            )

        message = str(ctx.exception)
        self.assertNotIn("INSERT", message)
        self.assertNotIn("UPDATE", message)
        self.assertNotIn(self.db_path, message)
        self.assertNotIn(str(Path(self.db_path).parent), message)


class _FailAgentRunConnection:
    """代理连接：对 agent_runs 插入抛错，验证 finalize 事务回滚。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql, params=None):
        if "INSERT INTO agent_runs" in str(sql):
            raise sqlite3.OperationalError("模拟 agent_run 插入失败")
        return self._conn.execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


@contextmanager
def _fail_agent_run_connection(path):
    with _REAL_CONNECTION(path) as conn:
        yield _FailAgentRunConnection(conn)


class FinalizeConversationTurnTests(RepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.conversation = create_conversation("轮次保存测试", path=self.db_path)
        self.user_message = insert_message(
            {
                "conversation_id": self.conversation["id"],
                "role": "user",
                "display_content": "题目",
                "model_content": "题目",
            },
            path=self.db_path,
        )

    def test_finalize_saves_assistant_run_and_state_atomically(self) -> None:
        finalized = finalize_conversation_turn(
            self.conversation["id"],
            assistant_message={
                "conversation_id": self.conversation["id"],
                "role": "assistant",
                "display_content": "回答",
                "model_content": "回答",
            },
            agent_run={
                "conversation_id": self.conversation["id"],
                "user_message_id": self.user_message["id"],
                "status": "completed",
                "teaching_mode": "hint",
                "use_rag": False,
                "use_tools": True,
                "total_model_requests": 2,
                "total_duration_ms": 10.5,
                "trace_json": {"run_id": "run-finalize"},
            },
            conversation_state={
                "active_problem_text": "求电流",
                "teaching_mode": "hint",
                "hint_step": 1,
            },
            path=self.db_path,
        )

        messages = list_messages(self.conversation["id"], path=self.db_path)
        runs = get_agent_runs(self.conversation["id"], path=self.db_path)
        state = get_conversation_state(self.conversation["id"], path=self.db_path)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["id"], finalized["assistant_message_id"])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["assistant_message_id"], finalized["assistant_message_id"])
        self.assertEqual(runs[0]["status"], "completed")
        self.assertEqual(runs[0]["trace_json"]["run_id"], "run-finalize")
        self.assertEqual(state["hint_step"], 1)
        self.assertEqual(state["teaching_mode"], "hint")

    def test_finalize_rolls_back_all_when_agent_run_insert_fails(self) -> None:
        with mock.patch(
            "src.storage.repositories._connection",
            _fail_agent_run_connection,
        ):
            with self.assertRaises(RepositoryError):
                finalize_conversation_turn(
                    self.conversation["id"],
                    assistant_message={
                        "conversation_id": self.conversation["id"],
                        "role": "assistant",
                        "display_content": "回答",
                        "model_content": "回答",
                    },
                    agent_run={
                        "conversation_id": self.conversation["id"],
                        "user_message_id": self.user_message["id"],
                        "status": "completed",
                        "use_rag": False,
                        "use_tools": False,
                        "total_model_requests": 2,
                        "total_duration_ms": 10.0,
                    },
                    path=self.db_path,
                )

        messages = list_messages(self.conversation["id"], path=self.db_path)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(
            get_agent_runs(self.conversation["id"], path=self.db_path),
            [],
        )
        self.assertIsNone(
            get_conversation_state(self.conversation["id"], path=self.db_path)
        )


class ClearConversationContentsTests(RepositoryTestCase):
    def test_clear_keeps_conversation_and_clears_children_only(self) -> None:
        conversation = create_conversation("清空测试", path=self.db_path)
        other = create_conversation("其他会话", path=self.db_path)
        insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "user",
                "display_content": "题目",
                "model_content": "题目",
            },
            path=self.db_path,
        )
        insert_message(
            {
                "conversation_id": other["id"],
                "role": "user",
                "display_content": "别动我",
                "model_content": "别动我",
            },
            path=self.db_path,
        )
        insert_agent_run(
            {
                "conversation_id": conversation["id"],
                "user_message_id": "user-x",
                "status": "completed",
                "use_rag": False,
                "use_tools": False,
                "total_model_requests": 1,
                "total_duration_ms": 1.0,
            },
            path=self.db_path,
        )
        upsert_conversation_state(
            {"conversation_id": conversation["id"], "hint_step": 2},
            path=self.db_path,
        )

        self.assertTrue(
            clear_conversation_contents(conversation["id"], path=self.db_path)
        )

        self.assertEqual(
            list_messages(conversation["id"], path=self.db_path),
            [],
        )
        self.assertEqual(
            get_agent_runs(conversation["id"], path=self.db_path),
            [],
        )
        self.assertIsNone(
            get_conversation_state(conversation["id"], path=self.db_path)
        )
        self.assertIsNotNone(get_conversation(conversation["id"], path=self.db_path))
        self.assertEqual(len(list_messages(other["id"], path=self.db_path)), 1)

    def test_clear_missing_conversation_returns_false(self) -> None:
        self.assertFalse(
            clear_conversation_contents("missing", path=self.db_path)
        )


if __name__ == "__main__":
    unittest.main()
