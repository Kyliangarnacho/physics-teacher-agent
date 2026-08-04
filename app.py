"""初中物理教师 Agent 的 Streamlit 聊天页面。"""

import streamlit as st

from src import agent as agent_module
from src.ui.paste_images import decode_pasted_images
from src.vision.batch import (
    build_batch_image_context,
    image_is_unreadable,
    image_needs_confirmation,
    merge_image_inputs,
    process_image_batch,
)
from src.vision.context import build_image_context_draft


MODEL_ERROR_MESSAGES = {
    "认证失败：请检查 DASHSCOPE_API_KEY 是否正确。",
    "网络连接失败：请检查网络或 QWEN_BASE_URL。",
    "请求过于频繁：请稍后再试。",
    "千问 API 请求失败，请稍后再试。",
    "调用千问时发生未知错误，请稍后再试。",
    "千问返回了空回答。",
}

TEACHING_MODE_OPTIONS = {
    "auto": "自动判断",
    "solve": "完整解题",
    "explain": "概念讲解",
    "hint": "只给提示",
    "diagnose": "错误诊断",
}

RAG_POLICY_OPTIONS = {
    "auto": "自动决定",
    "force": "强制使用",
    "off": "不使用",
}

VISION_MODE_OPTIONS = {
    "auto": "自动判断",
    "vision": "仅综合视觉",
    "ocr_enhanced": "OCR 增强",
}

PASTE_BRIDGE_KEY = "paste_image_bridge"
DEFAULT_IMAGE_QUESTION = "请分析并解答这些图片中的物理问题。"


APP_STYLES = """
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 800px;
    padding-top: 4.5rem;
    padding-bottom: 7rem;
}

.st-key-empty_state {
    width: min(100%, 650px);
    min-height: calc(100vh - 19rem);
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-inline: auto;
    padding-bottom: 12vh;
}

.st-key-empty_state h2 {
    margin: 0 0 1.65rem;
    color: #202123;
    font-size: clamp(1.65rem, 3vw, 2rem);
    font-weight: 500;
    letter-spacing: -0.035em;
    text-align: center;
}

[data-testid="stChatInput"] {
    width: min(100%, 650px);
    margin-inline: auto;
    border: 1px solid #dedede;
    border-radius: 999px !important;
    background: #ffffff;
    box-shadow: 0 6px 24px rgb(0 0 0 / 7%);
    overflow: hidden;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    border-radius: 999px !important;
    background: #ffffff;
}

[data-testid="stChatInput"] textarea {
    padding-left: 0.35rem;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #c7c7c7;
    box-shadow: 0 8px 28px rgb(0 0 0 / 9%);
}

.st-key-paste_image_bridge {
    width: min(calc(100vw - 2rem), 650px);
    margin: 0 auto 0.45rem;
}

[data-testid="stAppViewContainer"]:has(.st-key-empty_state)
    [data-testid="stBottom"] {
    transform: translateY(-44vh);
    background: transparent;
}

[data-testid="stExpander"] {
    background: #fafafa;
    border-color: #e7e7e7;
    border-radius: 0.75rem;
    margin-top: 0.8rem;
}

[data-testid="stAlert"] {
    border-radius: 0.75rem;
}

section[data-testid="stSidebar"] [data-testid="stButton"] button {
    width: 100%;
}

section[data-testid="stSidebar"] hr {
    margin: 1.25rem 0;
}

section[data-testid="stSidebar"] {
    color: #303030;
}

[class*="st-key-conversation_preview_"] {
    color: #5f6368;
    font-size: 0.9rem;
    line-height: 1.45;
}

[class*="st-key-user_message_row_"],
[class*="st-key-assistant_message_row_"] {
    margin: 0.45rem 0 1rem;
}

[class*="st-key-user_message_bubble_"],
[class*="st-key-assistant_message_bubble_"] {
    width: fit-content !important;
    min-height: 3rem;
    padding: 0.75rem 1rem;
    line-height: 1.65;
    word-break: normal;
    overflow-wrap: anywhere;
}

[class*="st-key-user_message_bubble_"] {
    min-width: 3.5rem;
    max-width: min(78%, 38rem);
    margin-left: auto;
    background: #e7e7e9;
    border-radius: 1.25rem 1.25rem 0.35rem 1.25rem;
}

[class*="st-key-assistant_message_bubble_"] {
    width: 100% !important;
    min-width: 0;
    max-width: 42rem;
    min-height: 0;
    margin-right: auto;
    padding: 0.35rem 0.15rem;
    background: transparent;
    border: 0;
    border-radius: 0;
}

[class*="st-key-user_message_bubble_"] [data-testid="stMarkdownContainer"],
[class*="st-key-assistant_message_bubble_"] [data-testid="stMarkdownContainer"] {
    margin: 0;
    padding: 0;
}

[class*="st-key-user_message_bubble_"] p,
[class*="st-key-assistant_message_bubble_"] p {
    margin-top: 0;
}

[class*="st-key-user_message_bubble_"] p:last-child,
[class*="st-key-assistant_message_bubble_"] p:last-child {
    margin-bottom: 0 !important;
}

@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] {
        max-width: 100%;
        padding: 3.75rem 0.8rem 6.25rem;
    }

    .st-key-empty_state {
        min-height: calc(100dvh - 17rem);
        padding-bottom: 15vh;
    }

    .st-key-empty_state h2 {
        margin-bottom: 1.25rem;
        font-size: 1.55rem;
    }

    [class*="st-key-user_message_bubble_"] {
        max-width: 86%;
    }

    [class*="st-key-assistant_message_bubble_"] {
        max-width: 100%;
    }

    .st-key-paste_image_bridge {
        width: calc(100vw - 1.6rem);
    }
}
</style>
"""


def clear_conversation() -> None:
    """Clear chat plus unsent image work, without deleting the safe cache."""
    st.session_state.messages = []
    for key in (
        "pending_chat_submission",
        "pending_submission",
        PASTE_BRIDGE_KEY,
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if str(key).startswith("pending_image_draft_"):
            st.session_state.pop(key, None)
    st.session_state.paste_bridge_reset_token = (
        int(st.session_state.get("paste_bridge_reset_token", 0)) + 1
    )


def queue_chat_submission(input_key: str) -> None:
    """Copy one native chat submission and pending pasted bytes for processing."""
    value = st.session_state.get(input_key)
    if isinstance(value, str):
        text = value
        files = []
    else:
        text = getattr(value, "text", "")
        files = getattr(value, "files", [])

    attached_images: list[dict[str, object]] = []
    if isinstance(files, list):
        for file in files:
            getvalue = getattr(file, "getvalue", None)
            filename = getattr(file, "name", "")
            if callable(getvalue) and isinstance(filename, str):
                attached_images.append(
                    {
                        "filename": filename,
                        "mime_type": str(getattr(file, "type", "")),
                        "bytes": getvalue(),
                    }
                )

    pasted_images = decode_pasted_images(
        st.session_state.get(PASTE_BRIDGE_KEY)
    )
    if str(text).strip() or pasted_images or attached_images:
        st.session_state.pending_chat_submission = {
            "text": str(text),
            "pasted_images": pasted_images,
            "attached_images": attached_images,
        }
    st.session_state.pop(PASTE_BRIDGE_KEY, None)
    st.session_state.paste_bridge_reset_token = (
        int(st.session_state.get("paste_bridge_reset_token", 0)) + 1
    )


def render_batch_details(batch: dict[str, object]) -> None:
    """Show safe visual metadata and traces in a collapsed area."""
    images = batch.get("images", [])
    if not isinstance(images, list) or not images:
        return
    with st.expander(
        "题图识别详情",
        expanded=False,
        icon=":material/image_search:",
    ):
        for position, record in enumerate(images, start=1):
            if not isinstance(record, dict):
                continue
            if position > 1:
                st.divider()
            extraction = record.get("extraction")
            status = getattr(getattr(extraction, "status", ""), "value", "")
            image_type = getattr(
                getattr(extraction, "image_type", ""),
                "value",
                "",
            )
            st.markdown(
                f"**图片 {record.get('index', position)} · "
                f"{record.get('filename', '')}**  \n"
                f"状态：`{status}` · 类型：`{image_type}` · "
                f"OCR：`{'已使用' if record.get('ocr_used') else '未使用'}` · "
                f"模型请求：`{record.get('model_requests', 0)}`"
            )
            traces = record.get("vision_step_traces", [])
            if isinstance(traces, list):
                for step in traces:
                    if not isinstance(step, dict):
                        continue
                    st.caption(
                        f"{step.get('name', '')}: {step.get('status', '')}, "
                        f"attempts={step.get('attempts', 0)}, "
                        f"model_requests={step.get('model_requests', 0)}"
                    )


def render_rag_sources(message: dict[str, object]) -> None:
    """重新渲染一次 RAG 回答的来源或无来源提示。"""
    route = message.get("route")
    if isinstance(route, dict):
        rag_enabled = bool(route.get("use_rag", False))
    else:
        rag_enabled = bool(message.get("rag_enabled", False))

    if not rag_enabled:
        return

    sources = message.get("sources", [])
    if not isinstance(sources, list) or not sources:
        st.info("本地知识库未检索到相关资料，本次按普通问答处理。")
        return

    with st.expander(
        "本地知识库参考来源",
        expanded=False,
        icon=":material/library_books:",
    ):
        for source in sources:
            if not isinstance(source, dict):
                continue
            st.markdown(
                f"**{source.get('id', '')} · {source.get('topic', '')}**  \n"
                f"来源：{source.get('source', '')}"
            )


def render_agent_decision(message: dict[str, object]) -> None:
    """以简洁字段展示一次 Agent 的分析与路由决策。"""
    analysis = message.get("analysis")
    route = message.get("route")
    if not isinstance(analysis, dict) or not isinstance(route, dict):
        return

    teaching_mode = str(route.get("teaching_mode", ""))
    mode_label = TEACHING_MODE_OPTIONS.get(
        teaching_mode,
        teaching_mode or "未知",
    )
    use_rag = "是" if route.get("use_rag", False) else "否"
    calculation_required = (
        "是" if analysis.get("calculation_required", False) else "否"
    )
    use_tools = "是" if route.get("use_tools", False) else "否"
    fallback = "是" if message.get("analysis_fallback", False) else "否"

    with st.expander(
        "Agent 决策",
        expanded=False,
        icon=":material/account_tree:",
    ):
        st.markdown(
            f"**最终教学模式：** {mode_label}  \n"
            f"**物理主题：** {analysis.get('physics_topic', '未知')}  \n"
            f"**问题类型：** {analysis.get('question_type', '未知')}  \n"
            f"**是否使用知识库：** {use_rag}  \n"
            f"**Analyzer 判断需要计算：** {calculation_required}  \n"
            f"**Router 启用本地工具：** {use_tools}  \n"
            f"**简短判断理由：** {analysis.get('short_reason', '无')}  \n"
            f"**Analyzer 是否发生 fallback：** {fallback}"
        )


def _tool_record_to_dict(record: object) -> dict[str, object] | None:
    """将字典或 Pydantic 工具记录转为统一的可展示字典。"""
    if isinstance(record, dict):
        return record

    model_dump = getattr(record, "model_dump", None)
    if not callable(model_dump):
        return None

    dumped = model_dump(mode="json")
    return dumped if isinstance(dumped, dict) else None


def render_tool_records(message: dict[str, object]) -> None:
    """渲染非空的本地计算工具执行记录。"""
    records = message.get("tool_records")
    if not isinstance(records, list) or not records:
        return

    display_records = [
        record_data
        for record in records
        if (record_data := _tool_record_to_dict(record)) is not None
    ]
    if not display_records:
        return

    with st.expander(
        "本地计算工具记录",
        expanded=False,
        icon=":material/calculate:",
    ):
        st.caption(
            "Tool Client 内部模型请求数："
            f"{message.get('tool_model_requests', 0)}"
        )
        for index, record in enumerate(display_records, start=1):
            if index > 1:
                st.divider()

            status = record.get("status", "")
            status_text = getattr(status, "value", status)
            st.markdown(
                f"**记录 {index}**  \n"
                f"**name：** `{record.get('name', '')}`  \n"
                f"**status：** `{status_text}`"
            )

            st.markdown("**arguments：**")
            arguments = record.get("arguments", {})
            st.json(arguments if isinstance(arguments, dict) else {})

            normalized_fields = record.get("normalized_fields", [])
            if isinstance(normalized_fields, list) and normalized_fields:
                normalized_text = ", ".join(
                    str(field) for field in normalized_fields
                )
            else:
                normalized_text = "无"
            st.markdown(f"**normalized_fields：** {normalized_text}")

            result = record.get("result")
            if isinstance(result, dict):
                display_value = result.get("display_value", "")
                unit = result.get("unit", "")
                st.markdown(
                    f"**formula：** `{result.get('formula', '')}`  \n"
                    f"**result：** `{display_value} {unit}`"
                )
            else:
                st.markdown("**result：** 无")

            error = record.get("error")
            if error:
                st.error(str(error))
            else:
                st.markdown("**error：** 无")


def render_run_trace(message: dict[str, object]) -> None:
    """展示一次 Agent 运行的安全摘要与步骤级观测字段。"""
    trace = message.get("trace")
    if not isinstance(trace, dict):
        return

    with st.expander(
        "Agent 运行轨迹",
        expanded=False,
        icon=":material/timeline:",
    ):
        st.markdown(
            f"**run_id：** `{trace.get('run_id', '')}`  \n"
            f"**status：** `{trace.get('status', '')}`  \n"
            f"**total_duration_ms：** `{trace.get('total_duration_ms', 0)}`  \n"
            f"**total_model_requests：** `{trace.get('total_model_requests', 0)}`  \n"
            f"**rag_searches：** `{trace.get('rag_searches', 0)}`  \n"
            f"**tool_executions：** `{trace.get('tool_executions', 0)}`  \n"
            f"**analysis_fallback：** `{trace.get('analysis_fallback', False)}`"
        )

        steps = trace.get("steps", [])
        if not isinstance(steps, list):
            return
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            st.divider()
            st.markdown(
                f"**步骤 {index} · {step.get('name', '')}**  \n"
                f"**status：** `{step.get('status', '')}`  \n"
                f"**attempts：** `{step.get('attempts', 0)}`  \n"
                f"**duration_ms：** `{step.get('duration_ms', 0)}`  \n"
                f"**model_requests：** `{step.get('model_requests', 0)}`"
            )
            error_type = step.get("error_type")
            error_message = step.get("error_message")
            if error_type or error_message:
                st.markdown(
                    f"**error_type：** `{error_type or ''}`  \n"
                    f"**error_message：** {error_message or ''}"
                )


def render_chat_message(message: dict[str, object], message_index: int) -> None:
    """使用无头像的左右气泡渲染一条聊天消息。"""
    role = str(message.get("role", "assistant"))
    is_user = role == "user"
    alignment = "right" if is_user else "left"
    row_key = f"{role}_message_row_{message_index}"
    bubble_key = f"{role}_message_bubble_{message_index}"

    with st.container(key=row_key, horizontal_alignment=alignment):
        with st.container(key=bubble_key, width="stretch"):
            st.markdown(str(message.get("content", "")))
            if not is_user:
                render_rag_sources(message)
                render_agent_decision(message)
                render_tool_records(message)
                render_run_trace(message)


def _image_history_metadata(
    batch: dict[str, object] | None,
    *,
    context_used: bool,
) -> dict[str, object]:
    """Build history-safe image metadata without bytes or extracted context."""
    if not isinstance(batch, dict):
        return {
            "image_count": 0,
            "image_filenames": [],
            "image_hash": [],
            "batch_id": None,
            "image_context_used": False,
            "vision_run_id": [],
        }
    images = batch.get("images", [])
    records = images if isinstance(images, list) else []
    return {
        "image_count": len(records),
        "image_filenames": [
            str(record.get("filename", ""))
            for record in records
            if isinstance(record, dict)
        ],
        "image_hash": [
            str(record.get("image_hash", ""))
            for record in records
            if isinstance(record, dict)
        ],
        "batch_id": batch.get("batch_id"),
        "image_context_used": context_used,
        "vision_run_id": [
            str(record.get("vision_run_id", ""))
            for record in records
            if isinstance(record, dict)
        ],
    }


def _pending_draft_key(batch_id: str, image_index: int) -> str:
    return f"pending_image_draft_{batch_id}_{image_index}"


def _clear_pending_submission() -> None:
    pending = st.session_state.pop("pending_submission", None)
    if not isinstance(pending, dict):
        return
    batch = pending.get("batch")
    if not isinstance(batch, dict):
        return
    batch_id = str(batch.get("batch_id", ""))
    images = batch.get("images", [])
    if isinstance(images, list):
        for record in images:
            if isinstance(record, dict) and isinstance(record.get("index"), int):
                st.session_state.pop(
                    _pending_draft_key(batch_id, record["index"]),
                    None,
                )


def _store_pending_submission(
    question: str,
    batch: dict[str, object],
) -> None:
    """Store only safe extraction results and editable drafts, never bytes."""
    st.session_state.pending_submission = {
        "question": question,
        "batch": batch,
    }
    batch_id = str(batch.get("batch_id", ""))
    images = batch.get("images", [])
    if not isinstance(images, list):
        return
    for record in images:
        if not isinstance(record, dict) or not image_needs_confirmation(record):
            continue
        extraction = record.get("extraction")
        index = record.get("index")
        if isinstance(index, int):
            st.session_state[
                _pending_draft_key(batch_id, index)
            ] = build_image_context_draft(extraction)


def render_pending_confirmation() -> dict[str, object] | None:
    """Render only ambiguous images and return a confirmed Agent payload."""
    pending = st.session_state.get("pending_submission")
    if not isinstance(pending, dict):
        return None
    question = pending.get("question")
    batch = pending.get("batch")
    if not isinstance(question, str) or not isinstance(batch, dict):
        return None
    images = batch.get("images", [])
    if not isinstance(images, list):
        return None

    st.warning("部分题图需要你核对。确认前不会调用教师 Agent。")
    render_batch_details(batch)
    batch_id = str(batch.get("batch_id", ""))
    confirmation_records = [
        record
        for record in images
        if isinstance(record, dict) and image_needs_confirmation(record)
    ]
    for record in confirmation_records:
        index = record.get("index")
        filename = record.get("filename", "")
        if not isinstance(index, int):
            continue
        with st.expander(
            f"核对图片 {index}：{filename}",
            expanded=True,
            icon=":material/edit_note:",
        ):
            st.text_area(
                "可编辑图片上下文",
                key=_pending_draft_key(batch_id, index),
                height=220,
            )

    if st.button(
        "确认并发送",
        key=f"confirm_batch_{batch_id}",
        type="primary",
        width="stretch",
    ):
        overrides: dict[int, str] = {}
        for record in confirmation_records:
            index = record.get("index")
            if not isinstance(index, int):
                continue
            draft = str(
                st.session_state.get(
                    _pending_draft_key(batch_id, index),
                    "",
                )
            ).strip()
            if not draft:
                st.warning(f"图片 {index} 的确认内容不能为空。")
                return None
            overrides[index] = draft
        return {
            "question": question,
            "batch": batch,
            "image_context": build_batch_image_context(
                images,
                context_overrides=overrides,
            ),
        }
    return None


st.set_page_config(
    page_title="初中物理教师 Agent",
    page_icon=":material/science:",
    layout="centered",
    initial_sidebar_state="auto",
)
st.html(APP_STYLES)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vision_batch_cache", {})
st.session_state.setdefault("paste_bridge_reset_token", 0)
queued_submission = st.session_state.pop("pending_chat_submission", None)
submission_to_process = None
submission_error = None
mode_override_for_question = str(
    st.session_state.get("teaching_mode", "auto")
)
rag_policy_for_question = str(
    st.session_state.get("rag_policy", "auto")
)

if isinstance(queued_submission, dict):
    try:
        queued_images = merge_image_inputs(
            queued_submission.get("pasted_images", []),
            queued_submission.get("attached_images", []),
        )
    except ValueError as exc:
        submission_error = str(exc)
    else:
        submitted_text = str(queued_submission.get("text", "")).strip()
        if submitted_text or queued_images:
            normalized_question = submitted_text or DEFAULT_IMAGE_QUESTION
            submission_to_process = {
                "question": normalized_question,
                "images": queued_images,
            }
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": normalized_question,
                    "image_count": len(queued_images),
                    "image_filenames": [
                        str(item["filename"]) for item in queued_images
                    ],
                    "image_hash": [
                        str(item["raw_hash"]) for item in queued_images
                    ],
                    "batch_id": None,
                    "image_context_used": False,
                    "vision_run_id": [],
                }
            )

with st.sidebar:
    st.markdown("**初中物理教师**")
    st.caption(
        "面向初中物理学习的个人教师 Agent，支持普通问答、"
        "本地知识库辅助与本地确定性计算工具。"
    )
    st.button(
        "清空当前对话",
        key="clear_conversation",
        icon=":material/edit_square:",
        width="stretch",
        on_click=clear_conversation,
    )

    st.markdown("##### 当前对话")
    user_message_previews = [
        str(message.get("content", "")).strip()
        for message in st.session_state.messages
        if message.get("role") == "user"
    ]
    if user_message_previews:
        for preview_index, preview in enumerate(
            reversed(user_message_previews[-3:])
        ):
            preview_text = preview if len(preview) <= 24 else f"{preview[:24]}…"
            with st.container(key=f"conversation_preview_{preview_index}"):
                st.markdown(f":material/chat_bubble_outline: {preview_text}")
    else:
        st.caption("还没有对话，先问一道题吧。")

    st.divider()
    st.markdown("##### 回答设置")
    teaching_mode = st.selectbox(
        "教学模式",
        options=list(TEACHING_MODE_OPTIONS),
        index=0,
        format_func=TEACHING_MODE_OPTIONS.get,
        key="teaching_mode",
        help="自动判断会根据问题内容选择完整解题、概念讲解、提示或错误诊断。",
    )
    rag_policy = st.selectbox(
        "知识库策略",
        options=list(RAG_POLICY_OPTIONS),
        index=0,
        format_func=RAG_POLICY_OPTIONS.get,
        key="rag_policy",
        help="自动决定由 Analyzer 判断；强制使用会检索本地知识库；不使用则直接回答。",
    )
    st.caption(
        f"当前：{TEACHING_MODE_OPTIONS[teaching_mode]} · "
        f"{RAG_POLICY_OPTIONS[rag_policy]}"
    )

    st.divider()
    st.markdown("##### 题图识别设置")
    vision_mode = st.selectbox(
        "识别模式",
        options=list(VISION_MODE_OPTIONS),
        format_func=VISION_MODE_OPTIONS.get,
        key="vision_mode",
    )
    vision_instruction = st.text_input(
        "识别重点（可选）",
        key="vision_instruction",
        placeholder="例如：重点核对电路连接和电表量程",
    )
    st.caption("在聊天框粘贴或选择图片，发送后才会开始识别。")

is_empty_state = (
    not st.session_state.messages
    and submission_to_process is None
    and st.session_state.get("pending_submission") is None
)
response_slot = None

if is_empty_state:
    with st.container(key="empty_state"):
        st.markdown("## 今天想从哪道物理题开始？")
        if submission_error:
            st.warning(submission_error)
else:
    for message_index, message in enumerate(st.session_state.messages):
        render_chat_message(message, message_index)

    response_slot = st.empty()

pending_agent_payload = None
if st.session_state.get("pending_submission") is not None:
    pending_agent_payload = render_pending_confirmation()

st.chat_input(
    "把题目或困惑发过来"
    if is_empty_state
    else "继续问一道物理问题…",
    key="question_input",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=8,
    submit_mode="disable",
    disabled=st.session_state.get("pending_submission") is not None,
    on_submit=queue_chat_submission,
    args=("question_input",),
)

agent_payload = pending_agent_payload
if submission_error and response_slot is not None:
    with response_slot.container():
        st.warning(submission_error)

if submission_to_process is not None:
    question = str(submission_to_process["question"])
    images = submission_to_process["images"]
    if images:
        try:
            with st.status(
                f"正在读取 {len(images)} 张题图",
                expanded=False,
            ) as image_status:
                batch = process_image_batch(
                    images,
                    mode=vision_mode,
                    user_instruction=vision_instruction,
                    cache=st.session_state.vision_batch_cache,
                    progress_func=lambda index, total: st.write(
                        f"正在识别图片 {index}/{total}"
                    ),
                )
                image_status.update(label="题图识别完成", state="complete")
        except Exception:
            if response_slot is not None:
                with response_slot.container():
                    st.error("题图处理失败，请检查图片后重新发送。")
        else:
            batch_images = batch.get("images", [])
            if any(image_is_unreadable(record) for record in batch_images):
                if response_slot is not None:
                    with response_slot.container():
                        st.error("有题图无法识别，请移除或替换后重新发送。")
                        render_batch_details(batch)
            elif any(
                image_needs_confirmation(record) for record in batch_images
            ):
                _store_pending_submission(question, batch)
                if response_slot is not None:
                    with response_slot.container():
                        render_pending_confirmation()
            else:
                render_batch_details(batch)
                agent_payload = {
                    "question": question,
                    "batch": batch,
                    "image_context": build_batch_image_context(batch_images),
                }
    else:
        agent_payload = {
            "question": question,
            "batch": None,
            "image_context": None,
        }

if agent_payload is not None:
    question = str(agent_payload["question"])
    batch = agent_payload.get("batch")
    image_context = agent_payload.get("image_context")
    try:
        with st.spinner("正在生成讲解……"):
            if isinstance(image_context, str) and image_context.strip():
                agent_result = agent_module.run_teacher_agent(
                    question,
                    mode_override=mode_override_for_question,
                    rag_policy=rag_policy_for_question,
                    image_context=image_context,
                    image_context_available=True,
                )
            else:
                agent_result = agent_module.run_teacher_agent(
                    question,
                    mode_override=mode_override_for_question,
                    rag_policy=rag_policy_for_question,
                )
            answer = agent_result["answer"]
    except Exception:
        if response_slot is not None:
            with response_slot.container():
                st.error("回答生成失败，请稍后再试。")
    else:
        if answer in MODEL_ERROR_MESSAGES:
            if response_slot is not None:
                with response_slot.container():
                    st.error(answer)
        else:
            metadata = _image_history_metadata(
                batch if isinstance(batch, dict) else None,
                context_used=isinstance(image_context, str),
            )
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": agent_result.get("sources", []),
                "analysis": agent_result.get("analysis", {}),
                "route": agent_result.get("route", {}),
                "analysis_fallback": agent_result.get(
                    "analysis_fallback",
                    False,
                ),
                "tool_records": agent_result.get("tool_records", []),
                "tool_model_requests": agent_result.get(
                    "tool_model_requests",
                    0,
                ),
                "trace": agent_result.get("trace"),
                **metadata,
            }
            if isinstance(batch, dict):
                batch_metadata = _image_history_metadata(
                    batch,
                    context_used=True,
                )
                for message in reversed(st.session_state.messages):
                    if (
                        message.get("role") == "user"
                        and message.get("batch_id") is None
                    ):
                        message.update(batch_metadata)
                        break
            st.session_state.messages.append(assistant_message)
            _clear_pending_submission()
            if response_slot is not None:
                with response_slot.container():
                    render_chat_message(
                        assistant_message,
                        len(st.session_state.messages) - 1,
                    )
