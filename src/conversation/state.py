"""Stage 10 Conversation State 纯逻辑。

负责跟进表达识别、完整答案请求识别、状态解析、模型上下文文本与一轮结束后的
状态推进。本模块不连接数据库、不调用模型，所有输入输出均为纯 Python 对象。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.conversation.schemas import ConversationState, validate_safe_text


_FOLLOW_UP_MARKERS = (
    "继续",
    "下一步",
    "再提示",
    "那为什么",
    "刚才",
    "上一题",
    "这张图",
)

_FULL_ANSWER_MARKERS = (
    "直接告诉答案",
    "直接给答案",
    "告诉我答案",
    "完整解答",
    "不要提示",
    "不用提示",
    "别提示",
    "直接算完",
)


def is_follow_up_message(text: str) -> bool:
    """保守识别明确跟进表达。

    跟进标记必须出现在文本开头（例如“继续”“下一步”“刚才”“这张图”），
    避免把包含普通词语的完整新题目误判为跟进；空文本返回 False。
    """
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    return any(stripped.startswith(marker) for marker in _FOLLOW_UP_MARKERS)


def requests_full_answer(text: str) -> bool:
    """识别“直接告诉答案、完整解答、不要提示、直接算完”等退出逐步提示的表达。"""
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    return any(marker in stripped for marker in _FULL_ANSWER_MARKERS)


@dataclass(frozen=True)
class ResolvedConversationState:
    """解析后的会话状态快照，供模型上下文与状态推进使用。"""

    active_problem_text: str | None
    active_image_context: str | None
    teaching_mode: str | None
    hint_step: int = 0
    is_follow_up: bool = False
    exits_hint: bool = False
    conversation_id: str | None = None


def _as_conversation_state(
    previous_state: ConversationState | dict | None,
) -> ConversationState | None:
    if previous_state is None:
        return None
    if isinstance(previous_state, ConversationState):
        return previous_state
    return ConversationState.model_validate(previous_state)


def resolve_conversation_state(
    question: str,
    previous_state: ConversationState | dict | None = None,
    current_image_context: str | None = None,
    mode_override: str = "auto",
) -> ResolvedConversationState:
    """根据当前问题与旧状态解析新一轮会话状态。

    优先级：
    - 空问题抛 ValueError；
    - 明确跟进且旧状态存在时复用 active_problem_text，可复用旧安全图片上下文；
    - 新问题替换 active_problem_text，无新图片时清除旧图片上下文；
    - 新图片上下文优先于旧上下文；
    - 显式 mode_override 优先；“直接给答案”等请求退出 hint 并视为 solve；
    - 普通“继续/下一步”在旧状态为 hint 时继续 hint；
    - 不修改传入的 previous_state。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空。")
    question = question.strip()

    previous = _as_conversation_state(previous_state)
    image_context = validate_safe_text(current_image_context, "current_image_context")

    follow_up = is_follow_up_message(question)
    exits_hint = requests_full_answer(question)

    if follow_up and previous is not None:
        active_problem_text = previous.active_problem_text
    else:
        active_problem_text = question

    if image_context is not None:
        active_image_context = image_context
    elif follow_up and previous is not None:
        active_image_context = previous.active_image_context
    else:
        active_image_context = None

    if mode_override != "auto":
        effective_mode = mode_override
    elif exits_hint:
        effective_mode = "solve"
    elif follow_up and previous is not None and previous.teaching_mode == "hint":
        effective_mode = "hint"
    else:
        effective_mode = None

    hint_step = previous.hint_step if (follow_up and previous is not None) else 0

    return ResolvedConversationState(
        active_problem_text=active_problem_text,
        active_image_context=active_image_context,
        teaching_mode=effective_mode,
        hint_step=hint_step,
        is_follow_up=follow_up,
        exits_hint=exits_hint,
        conversation_id=previous.conversation_id if previous is not None else None,
    )


def build_teaching_state_context(
    resolved_state: ResolvedConversationState,
) -> str | None:
    """构建发送给模型的安全状态文本。

    只包含活动题目、已确认图片文字上下文、教学模式与已完成的提示步数；
    hint 模式下说明下一次从后续步骤继续。不包含历史回答、Trace、工具记录或
    内部思维过程。
    """
    if not resolved_state.active_problem_text:
        return None
    lines = [f"当前活动题目：{resolved_state.active_problem_text}"]
    if resolved_state.active_image_context:
        lines.append(
            f"已确认图片文字上下文：{resolved_state.active_image_context}"
        )
    if resolved_state.teaching_mode:
        lines.append(f"当前教学模式：{resolved_state.teaching_mode}")
    lines.append(f"已完成提示步数：{resolved_state.hint_step}")
    if resolved_state.teaching_mode == "hint":
        lines.append("下一步请从后续步骤继续，不要重复已完成步骤。")
    return "\n".join(lines)


def build_state_update_after_turn(
    resolved_state: ResolvedConversationState,
    *,
    succeeded: bool,
    effective_mode: str | None,
) -> dict | None:
    """计算一轮结束后的状态更新字段（可直接传给 upsert_conversation_state）。

    - 失败或 blocked（succeeded=False）时不推进，返回 None；
    - 成功且 effective_mode=hint 时 hint_step + 1；
    - 成功且非 hint 时 hint_step 重置为 0；
    - 不自行写数据库。
    """
    if not succeeded:
        return None

    hint_step = (
        resolved_state.hint_step + 1
        if effective_mode == "hint"
        else 0
    )
    update: dict[str, object] = {
        "active_problem_text": resolved_state.active_problem_text,
        "active_image_context": resolved_state.active_image_context,
        "hint_step": hint_step,
    }
    if effective_mode is not None:
        update["teaching_mode"] = effective_mode
    if resolved_state.conversation_id is not None:
        update["conversation_id"] = resolved_state.conversation_id
    return update
