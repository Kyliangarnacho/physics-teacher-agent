"""Stage 10 会话领域层：Schema 与最近历史窗口。

当前提供 ConversationRecord、StoredMessage、ConversationState 三个领域 Schema，
以及 build_recent_history 最近历史窗口；Conversation Service 将在后续任务实现。
"""

from src.conversation.history import build_recent_history
from src.conversation.schemas import (
    ConversationRecord,
    ConversationState,
    StoredMessage,
    validate_safe_text,
)
from src.conversation.state import (
    ResolvedConversationState,
    build_state_update_after_turn,
    build_teaching_state_context,
    is_follow_up_message,
    requests_full_answer,
    resolve_conversation_state,
)
from src.conversation.service import (
    ConversationServiceError,
    run_conversation_turn,
)

__all__ = [
    "ConversationServiceError",
    "ConversationRecord",
    "ConversationState",
    "ResolvedConversationState",
    "StoredMessage",
    "build_recent_history",
    "build_state_update_after_turn",
    "build_teaching_state_context",
    "is_follow_up_message",
    "requests_full_answer",
    "resolve_conversation_state",
    "run_conversation_turn",
    "validate_safe_text",
]
