"""SQLite Schema 迁移：使用 ``PRAGMA user_version`` 管理版本并保证原子性。

当前 schema 版本为 4，在既有会话数据上增加 conversation_summaries 表。
Agent 运行额外保存安全的 Analyzer 分析摘要，供历史恢复决策展示。字段名是后续
Repository 与 Conversation Service 的数据合同，
不得自行改名或用 JSON 大字段替代明确字段。每个版本在一个显式事务中应用，
成功时写入 ``user_version``，失败时整体回滚，不留下半成品结构。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence


CURRENT_SCHEMA_VERSION = 4


_SCHEMA_MIGRATIONS: dict[int, Sequence[str]] = {
    1: (
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL
                REFERENCES conversations(id)
                ON DELETE CASCADE,
            role TEXT NOT NULL,
            display_content TEXT NOT NULL,
            model_content TEXT NOT NULL,
            image_metadata_json TEXT,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE agent_runs (
            run_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL
                REFERENCES conversations(id)
                ON DELETE CASCADE,
            user_message_id TEXT NOT NULL,
            assistant_message_id TEXT,
            status TEXT NOT NULL,
            teaching_mode TEXT,
            use_rag INTEGER NOT NULL,
            use_tools INTEGER NOT NULL,
            total_model_requests INTEGER NOT NULL,
            total_duration_ms REAL NOT NULL,
            sources_json TEXT,
            tool_records_json TEXT,
            trace_json TEXT,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE conversation_states (
            conversation_id TEXT PRIMARY KEY
                REFERENCES conversations(id)
                ON DELETE CASCADE,
            active_problem_text TEXT,
            active_image_context TEXT,
            teaching_mode TEXT,
            hint_step INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE learning_memories (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            normalized_content TEXT NOT NULL,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            confidence REAL NOT NULL,
            source_conversation_id TEXT,
            source_message_id TEXT,
            confirmed INTEGER NOT NULL DEFAULT 1,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX idx_conversations_updated_at
            ON conversations(updated_at)
        """,
        """
        CREATE INDEX idx_messages_conversation_id
            ON messages(conversation_id)
        """,
        """
        CREATE INDEX idx_agent_runs_conversation_id
            ON agent_runs(conversation_id)
        """,
        """
        CREATE UNIQUE INDEX uq_learning_memories_dedup
            ON learning_memories(memory_type, topic, normalized_content)
        """,
        """
        CREATE INDEX idx_learning_memories_source_conversation_id
            ON learning_memories(source_conversation_id)
        """,
    ),
    2: (
        "ALTER TABLE agent_runs ADD COLUMN analysis_json TEXT",
    ),
    3: (
        """
        CREATE TABLE generation_jobs (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL
                REFERENCES conversations(id)
                ON DELETE CASCADE,
            user_message_id TEXT NOT NULL
                REFERENCES messages(id)
                ON DELETE CASCADE,
            status TEXT NOT NULL
                CHECK (status IN (
                    'pending', 'running', 'completed', 'failed', 'interrupted'
                )),
            payload_json TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0
                CHECK (attempts >= 0),
            error_type TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
        """,
        """
        CREATE INDEX idx_generation_jobs_conversation_id
            ON generation_jobs(conversation_id, created_at)
        """,
        """
        CREATE INDEX idx_generation_jobs_status_created_at
            ON generation_jobs(status, created_at)
        """,
        """
        CREATE UNIQUE INDEX uq_generation_jobs_active_conversation
            ON generation_jobs(conversation_id)
            WHERE status IN ('pending', 'running')
        """,
    ),
    4: (
        """
        CREATE TABLE conversation_summaries (
            conversation_id TEXT PRIMARY KEY
                REFERENCES conversations(id)
                ON DELETE CASCADE,
            summary_text TEXT NOT NULL,
            covered_until_message_id TEXT,
            covered_turn_count INTEGER NOT NULL DEFAULT 0,
            summary_revision INTEGER NOT NULL DEFAULT 1,
            model_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
    ),
}


def get_user_version(conn: sqlite3.Connection) -> int:
    """读取当前数据库 schema 版本（``PRAGMA user_version``）。"""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def apply_migration(
    conn: sqlite3.Connection,
    version: int,
    statements: Sequence[str],
) -> None:
    """在单个显式事务中应用一个 migration。

    成功时提交并将 ``user_version`` 更新为 ``version``；任何语句失败时回滚，
    保证不会留下半成品结构。
    """
    if not isinstance(version, int) or isinstance(version, bool):
        raise TypeError("version 必须是整数。")

    conn.execute("BEGIN IMMEDIATE")
    try:
        for statement in statements:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {version}")
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def migrate_database(conn: sqlite3.Connection) -> int:
    """将数据库升级到 ``CURRENT_SCHEMA_VERSION``。

    可重复执行：已是最新版本时直接返回；比代码版本更新时拒绝降级。
    每个版本单独提交，单个版本失败时整体回滚。
    """
    current = get_user_version(conn)
    if current > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"数据库 schema 版本 {current} 高于当前代码支持的 "
            f"{CURRENT_SCHEMA_VERSION}，拒绝降级。"
        )

    for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
        apply_migration(conn, version, _SCHEMA_MIGRATIONS[version])
    return get_user_version(conn)
