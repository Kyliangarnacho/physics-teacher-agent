"""Stage 10 最近历史窗口：从存储消息构建发送给模型的旧到新轮次。"""

from __future__ import annotations

from collections.abc import Sequence

from src.conversation.schemas import StoredMessage


def _as_stored_message(item: StoredMessage | dict) -> StoredMessage:
    if isinstance(item, StoredMessage):
        return item
    return StoredMessage.model_validate(item)


def _collect_complete_turns(
    messages: Sequence[StoredMessage],
) -> list[tuple[StoredMessage, StoredMessage]]:
    """按顺序识别“user 后紧跟 assistant”的完整轮次。

    只做相邻配对，不跨过其他角色重新拼轮次；不参与配对的孤立消息被忽略。
    """
    turns: list[tuple[StoredMessage, StoredMessage]] = []
    index = 0
    while index + 1 < len(messages):
        current = messages[index]
        following = messages[index + 1]
        if current.role == "user" and following.role == "assistant":
            turns.append((current, following))
            index += 2
        else:
            index += 1
    return turns


def _turn_chars(turn: tuple[StoredMessage, StoredMessage]) -> int:
    user_message, assistant_message = turn
    return len(user_message.model_content) + len(assistant_message.model_content)


def build_recent_history(
    messages: list[StoredMessage],
    max_turns: int = 3,
    max_chars: int = 6000,
) -> list[dict[str, str]]:
    """构建最近完整轮次的历史消息（旧到新），供模型上下文使用。

    规则：
    - 只保留 user 后紧跟 assistant 的完整轮次，忽略末尾未回答的 user；
    - 最多保留最近 max_turns 个完整轮次；
    - 字符预算按 model_content 长度计算，超预算时按完整轮次删除最旧内容；
    - 最新一个完整轮次本身超过预算时返回空列表；
    - 输出只包含 role/content（content 使用 model_content），不附带 Trace、
      来源、工具记录或图片原始数据；函数末尾不附加当前问题。
    """
    if not isinstance(max_turns, int) or not isinstance(max_chars, int):
        raise ValueError("max_turns 和 max_chars 必须是整数。")
    if max_turns < 0 or max_chars < 0:
        raise ValueError("max_turns 和 max_chars 不能为负数。")
    if max_turns == 0 or max_chars == 0:
        return []

    normalized = [_as_stored_message(item) for item in messages]
    turns = _collect_complete_turns(normalized)
    if not turns:
        return []

    turns = turns[-max_turns:]
    if _turn_chars(turns[-1]) > max_chars:
        return []

    selected = list(turns)
    total_chars = sum(_turn_chars(turn) for turn in selected)
    while selected and total_chars > max_chars:
        oldest = selected.pop(0)
        total_chars -= _turn_chars(oldest)

    history: list[dict[str, str]] = []
    for user_message, assistant_message in selected:
        history.append({"role": "user", "content": user_message.model_content})
        history.append(
            {"role": "assistant", "content": assistant_message.model_content}
        )
    return history
