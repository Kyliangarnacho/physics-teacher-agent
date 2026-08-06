"""Stage 10 本地 SQLite 存储层。

当前提供数据库路径解析、短生命周期连接、Schema Migration V1 与 Repository
数据访问层；Conversation Service 与长期记忆提取将在后续任务实现。
"""

from src.storage.database import (
    connect_database,
    get_database_path,
    initialize_database,
)
from src.storage.migrations import (
    CURRENT_SCHEMA_VERSION,
    apply_migration,
    get_user_version,
    migrate_database,
)
from src.storage.repositories import (
    RepositoryError,
    StorageError,
    clear_conversation_contents,
    create_conversation,
    deactivate_memory,
    delete_conversation,
    delete_memory,
    finalize_conversation_turn,
    get_agent_runs,
    get_conversation,
    get_conversation_state,
    get_recent_messages,
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

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "RepositoryError",
    "StorageError",
    "apply_migration",
    "clear_conversation_contents",
    "connect_database",
    "create_conversation",
    "deactivate_memory",
    "delete_conversation",
    "delete_memory",
    "finalize_conversation_turn",
    "get_agent_runs",
    "get_conversation",
    "get_conversation_state",
    "get_database_path",
    "get_recent_messages",
    "get_user_version",
    "insert_agent_run",
    "insert_message",
    "insert_or_merge_memory",
    "initialize_database",
    "list_conversations",
    "list_memories",
    "list_messages",
    "migrate_database",
    "rename_conversation",
    "reset_conversation_state",
    "upsert_conversation_state",
]
