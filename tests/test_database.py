"""Stage 10 SQLite 数据库初始化与 Migration V1 单元测试。"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from src.config import DEFAULT_DATABASE_PATH
from src.storage import (
    CURRENT_SCHEMA_VERSION,
    connect_database,
    get_database_path,
    initialize_database,
)
from src.storage import migrations


EXPECTED_TABLES = {
    "conversations",
    "messages",
    "agent_runs",
    "conversation_states",
    "learning_memories",
}

# PRAGMA table_info 每列返回 (type, notnull, dflt_value, pk) 的精确数据合同。
EXPECTED_COLUMNS = {
    "conversations": {
        "id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 1},
        "title": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "created_at": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "updated_at": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "archived": {"type": "INTEGER", "notnull": 1, "default": "0", "pk": 0},
    },
    "messages": {
        "id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 1},
        "conversation_id": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "role": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "display_content": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "model_content": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "image_metadata_json": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "created_at": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
    },
    "agent_runs": {
        "run_id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 1},
        "conversation_id": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "user_message_id": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "assistant_message_id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "status": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "teaching_mode": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "use_rag": {"type": "INTEGER", "notnull": 1, "default": None, "pk": 0},
        "use_tools": {"type": "INTEGER", "notnull": 1, "default": None, "pk": 0},
        "total_model_requests": {
            "type": "INTEGER",
            "notnull": 1,
            "default": None,
            "pk": 0,
        },
        "total_duration_ms": {"type": "REAL", "notnull": 1, "default": None, "pk": 0},
        "sources_json": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "tool_records_json": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "trace_json": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "analysis_json": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "created_at": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
    },
    "conversation_states": {
        "conversation_id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 1},
        "active_problem_text": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "active_image_context": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "teaching_mode": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "hint_step": {"type": "INTEGER", "notnull": 1, "default": "0", "pk": 0},
        "updated_at": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
    },
    "learning_memories": {
        "id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 1},
        "memory_type": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "topic": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "content": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "normalized_content": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "evidence_count": {"type": "INTEGER", "notnull": 1, "default": "1", "pk": 0},
        "confidence": {"type": "REAL", "notnull": 1, "default": None, "pk": 0},
        "source_conversation_id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "source_message_id": {"type": "TEXT", "notnull": 0, "default": None, "pk": 0},
        "confirmed": {"type": "INTEGER", "notnull": 1, "default": "1", "pk": 0},
        "active": {"type": "INTEGER", "notnull": 1, "default": "1", "pk": 0},
        "created_at": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
        "updated_at": {"type": "TEXT", "notnull": 1, "default": None, "pk": 0},
    },
}

# PRAGMA foreign_key_list 每行返回 (from, table, to, on_delete) 的精确数据合同。
EXPECTED_FOREIGN_KEYS = {
    "conversations": set(),
    "messages": {("conversation_id", "conversations", "id", "CASCADE")},
    "agent_runs": {("conversation_id", "conversations", "id", "CASCADE")},
    "conversation_states": {("conversation_id", "conversations", "id", "CASCADE")},
    "learning_memories": set(),
}

API_CONFIG_KEYS = (
    "DASHSCOPE_API_KEY",
    "QWEN_BASE_URL",
    "QWEN_MODEL",
    "QWEN_VISION_MODEL",
    "QWEN_OCR_MODEL",
)

ISO_TIME = "2026-08-06T08:00:00+00:00"


@contextmanager
def _env(name: str, value: str | None) -> Iterator[None]:
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row["name"] for row in rows}


def table_info(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, object]]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {
        row["name"]: {
            "type": row["type"],
            "notnull": row["notnull"],
            "default": row["dflt_value"],
            "pk": row["pk"],
        }
        for row in rows
    }


def foreign_key_specs(
    conn: sqlite3.Connection,
    table: str,
) -> set[tuple[str, str, str, str]]:
    rows = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    return {
        (row["from"], row["table"], row["to"], row["on_delete"])
        for row in rows
    }


def index_specs(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, object]]:
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    specs: dict[str, dict[str, object]] = {}
    for row in rows:
        name = row["name"]
        specs[name] = {
            "unique": bool(row["unique"]),
            "origin": row["origin"],
            "columns": [
                item["name"]
                for item in conn.execute(f"PRAGMA index_info({name})").fetchall()
            ],
        }
    return specs


class DatabasePathTests(unittest.TestCase):
    def test_default_path_is_stage10_default(self) -> None:
        with _env("PHYSICS_AGENT_DB_PATH", None):
            self.assertEqual(get_database_path(), DEFAULT_DATABASE_PATH)
            self.assertEqual(DEFAULT_DATABASE_PATH, "data/runtime/physics_teacher.db")

    def test_env_override_changes_database_path(self) -> None:
        override = "D:/custom/runtime/teacher.db"
        with _env("PHYSICS_AGENT_DB_PATH", override):
            self.assertEqual(get_database_path(), override)

    def test_database_path_does_not_require_api_config(self) -> None:
        saved = {name: os.environ.pop(name, None) for name in API_CONFIG_KEYS}
        try:
            with _env("PHYSICS_AGENT_DB_PATH", None):
                with mock.patch("src.config.load_dotenv", lambda: None):
                    path = get_database_path()
            self.assertEqual(path, DEFAULT_DATABASE_PATH)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class DatabaseConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(
            Path(self._temp_dir.name) / "nested" / "physics_teacher.db"
        )

    def test_connect_creates_parent_directories(self) -> None:
        conn = connect_database(self.db_path)
        conn.close()

        self.assertTrue(Path(self.db_path).parent.is_dir())
        self.assertTrue(Path(self.db_path).is_file())

    def test_connection_uses_expected_pragmas(self) -> None:
        conn = connect_database(self.db_path)
        try:
            self.assertIs(conn.row_factory, sqlite3.Row)
            self.assertEqual(
                conn.execute("PRAGMA journal_mode").fetchone()[0],
                "wal",
            )
            self.assertEqual(
                conn.execute("PRAGMA busy_timeout").fetchone()[0],
                5000,
            )
            self.assertEqual(
                conn.execute("PRAGMA foreign_keys").fetchone()[0],
                1,
            )
        finally:
            conn.close()

    def test_each_connect_returns_a_new_short_lived_connection(self) -> None:
        first = connect_database(self.db_path)
        second = connect_database(self.db_path)
        try:
            self.assertIsNot(first, second)
        finally:
            first.close()
            second.close()

    def test_rows_are_sqlite3_row_objects(self) -> None:
        conn = connect_database(self.db_path)
        try:
            conn.execute("CREATE TABLE sample (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO sample VALUES (1, '欧姆定律')")
            row = conn.execute("SELECT * FROM sample").fetchone()

            self.assertIsInstance(row, sqlite3.Row)
            self.assertEqual(row["id"], 1)
            self.assertEqual(row["name"], "欧姆定律")
        finally:
            conn.close()


class DatabaseMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "physics_teacher.db")

    def test_fresh_database_initializes_to_current_user_version(self) -> None:
        returned_path = initialize_database(self.db_path)

        self.assertEqual(returned_path, self.db_path)
        self.assertEqual(CURRENT_SCHEMA_VERSION, 2)
        conn = connect_database(self.db_path)
        try:
            self.assertEqual(migrations.get_user_version(conn), 2)
        finally:
            conn.close()

    def test_initialization_creates_five_tables(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            self.assertEqual(table_names(conn), EXPECTED_TABLES)
        finally:
            conn.close()

    def test_table_definitions_match_contract_exactly(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            for table, expected in EXPECTED_COLUMNS.items():
                with self.subTest(table=table):
                    self.assertEqual(table_info(conn, table), expected)
        finally:
            conn.close()

    def test_foreign_keys_match_contract(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            for table, expected in EXPECTED_FOREIGN_KEYS.items():
                with self.subTest(table=table):
                    self.assertEqual(foreign_key_specs(conn, table), expected)
        finally:
            conn.close()

    def test_foreign_key_is_enforced(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, role, "
                    "display_content, model_content, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("msg-bad", "missing-conv", "user", "内容", "内容", ISO_TIME),
                )
        finally:
            conn.close()

    def test_cascade_delete_conversation_removes_children(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("conv-1", "测试会话", ISO_TIME, ISO_TIME),
            )
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, "
                "display_content, model_content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("msg-1", "conv-1", "user", "题目", "题目", ISO_TIME),
            )
            conn.execute(
                "INSERT INTO agent_runs (run_id, conversation_id, user_message_id, "
                "status, use_rag, use_tools, total_model_requests, "
                "total_duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "run-1",
                    "conv-1",
                    "msg-1",
                    "completed",
                    0,
                    1,
                    3,
                    12.5,
                    ISO_TIME,
                ),
            )
            conn.execute(
                "INSERT INTO conversation_states (conversation_id, updated_at) "
                "VALUES (?, ?)",
                ("conv-1", ISO_TIME),
            )
            conn.commit()

            conn.execute("DELETE FROM conversations WHERE id = ?", ("conv-1",))

            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM conversation_states"
                ).fetchone()[0],
                0,
            )
        finally:
            conn.close()

    def test_default_values_are_applied_on_insert(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("conv-1", "会话", ISO_TIME, ISO_TIME),
            )
            conn.execute(
                "INSERT INTO conversation_states (conversation_id, updated_at) "
                "VALUES (?, ?)",
                ("conv-1", ISO_TIME),
            )
            conn.execute(
                "INSERT INTO learning_memories (id, memory_type, topic, content, "
                "normalized_content, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "mem-1",
                    "mistake",
                    "欧姆定律",
                    "学生常在单位上出错",
                    "欧姆定律单位错误",
                    0.9,
                    ISO_TIME,
                    ISO_TIME,
                ),
            )

            conversation = conn.execute(
                "SELECT archived FROM conversations WHERE id = ?",
                ("conv-1",),
            ).fetchone()
            state = conn.execute(
                "SELECT hint_step FROM conversation_states WHERE conversation_id = ?",
                ("conv-1",),
            ).fetchone()
            memory = conn.execute(
                "SELECT evidence_count, confirmed, active "
                "FROM learning_memories WHERE id = ?",
                ("mem-1",),
            ).fetchone()

            self.assertEqual(conversation["archived"], 0)
            self.assertEqual(state["hint_step"], 0)
            self.assertEqual(memory["evidence_count"], 1)
            self.assertEqual(memory["confirmed"], 1)
            self.assertEqual(memory["active"], 1)
        finally:
            conn.close()

    def test_learning_memories_unique_dedup_index(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            indexes = index_specs(conn, "learning_memories")
            self.assertIn("uq_learning_memories_dedup", indexes)
            self.assertTrue(indexes["uq_learning_memories_dedup"]["unique"])
            self.assertEqual(
                indexes["uq_learning_memories_dedup"]["columns"],
                ["memory_type", "topic", "normalized_content"],
            )

            conn.execute(
                "INSERT INTO learning_memories (id, memory_type, topic, content, "
                "normalized_content, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "mem-1",
                    "mistake",
                    "欧姆定律",
                    "内容一",
                    "欧姆定律单位错误",
                    0.9,
                    ISO_TIME,
                    ISO_TIME,
                ),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO learning_memories (id, memory_type, topic, "
                    "content, normalized_content, confidence, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "mem-2",
                        "mistake",
                        "欧姆定律",
                        "内容二",
                        "欧姆定律单位错误",
                        0.8,
                        ISO_TIME,
                        ISO_TIME,
                    ),
                )
        finally:
            conn.close()

    def test_required_query_indexes_exist(self) -> None:
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            self.assertIn(
                "idx_conversations_updated_at",
                index_specs(conn, "conversations"),
            )
            self.assertIn(
                "idx_messages_conversation_id",
                index_specs(conn, "messages"),
            )
            self.assertIn(
                "idx_agent_runs_conversation_id",
                index_specs(conn, "agent_runs"),
            )
            self.assertIn(
                "idx_learning_memories_source_conversation_id",
                index_specs(conn, "learning_memories"),
            )
        finally:
            conn.close()

    def test_initialization_is_repeatable(self) -> None:
        initialize_database(self.db_path)
        initialize_database(self.db_path)

        conn = connect_database(self.db_path)
        try:
            self.assertEqual(migrations.get_user_version(conn), 2)
            self.assertEqual(table_names(conn), EXPECTED_TABLES)
        finally:
            conn.close()

    def test_failed_migration_rolls_back_without_partial_schema(self) -> None:
        conn = connect_database(self.db_path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                migrations.apply_migration(
                    conn,
                    1,
                    (
                        "CREATE TABLE partial_table (id INTEGER)",
                        "THIS IS NOT VALID SQL",
                    ),
                )

            self.assertEqual(migrations.get_user_version(conn), 0)
            self.assertNotIn("partial_table", table_names(conn))

            self.assertEqual(migrations.migrate_database(conn), 2)
            self.assertEqual(table_names(conn), EXPECTED_TABLES)
        finally:
            conn.close()

    def test_version_1_database_migrates_to_analysis_summary_column(self) -> None:
        conn = connect_database(self.db_path)
        try:
            migrations.apply_migration(conn, 1, migrations._SCHEMA_MIGRATIONS[1])
            self.assertEqual(migrations.get_user_version(conn), 1)

            self.assertEqual(migrations.migrate_database(conn), 2)
            self.assertIn("analysis_json", table_info(conn, "agent_runs"))
        finally:
            conn.close()


class GitIgnoreTests(unittest.TestCase):
    def test_database_files_are_git_ignored(self) -> None:
        gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"
        content = gitignore_path.read_text(encoding="utf-8")
        lines = {line.strip() for line in content.splitlines()}

        for pattern in ("data/runtime/", "*.db", "*.db-shm", "*.db-wal"):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, lines)


if __name__ == "__main__":
    unittest.main()
