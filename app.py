"""初中物理教师 Agent 的 Streamlit 聊天页面。"""

import streamlit as st

from src.conversation import (
    ConversationServiceError,
    StoredMessage,
    build_recent_history,
    enqueue_conversation_turn,
)
from src.memory import (
    MemoryCandidate,
    MemoryServiceError,
    confirm_memory_candidate,
)
from src.storage import (
    AttachmentStoreError,
    clear_conversation_contents,
    create_conversation,
    deactivate_memory,
    delete_conversation,
    delete_conversation_attachments,
    delete_saved_attachments,
    delete_memory,
    get_active_generation_job,
    get_agent_runs,
    get_conversation,
    initialize_database,
    list_conversations,
    list_memories,
    list_messages,
    list_generation_jobs,
    rename_conversation,
    resolve_attachment_path,
    save_image_attachments,
)
from src.tasks import (
    GenerationTaskManager,
    MemoryAnalysisTaskError,
    MemoryAnalysisTaskManager,
    TaskManagerError,
)
from src.ui import editable_conversation_title, message_anchor
from src.ui.paste_images import decode_pasted_images
from src.vision.batch import (
    batch_needs_confirmation,
    build_batch_image_context,
    image_is_unreadable,
    image_needs_confirmation,
    merge_image_inputs,
    process_image_batch,
)
from src.vision.batch_relation import BatchRelation, BatchRelationAnalysis
from src.vision.context import build_image_context_draft


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

MEMORY_TYPE_LABELS = {
    "weakness": "薄弱点",
    "misconception": "错误观念",
    "preference": "讲解偏好",
}

MEMORY_SYSTEM_ERROR_PREFIXES = (
    "工具参数校验失败",
    "网络连接失败",
    "请求过于频繁",
    "千问 API 请求失败",
    "调用千问时发生未知错误",
    "千问返回了空回答",
    "请补充题图",
)

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
    border-radius: 1.35rem !important;
    background: #ffffff;
    box-shadow: 0 6px 24px rgb(0 0 0 / 7%);
    overflow: hidden;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background: #ffffff;
}

[data-testid="stChatInput"] > div {
    border-radius: inherit !important;
    align-items: flex-end;
}

[data-testid="stChatInput"] textarea {
    border-radius: 0 !important;
    min-height: 1.5rem !important;
    max-height: 8rem !important;
    padding: 0.65rem 0.35rem;
    line-height: 1.5;
    overflow-y: auto !important;
    resize: none !important;
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

.st-key-current_conversation_outline {
    max-height: 18rem;
    overflow-y: auto;
    padding-right: 0.25rem;
}

[class*="st-key-user_attachment_gallery_"] {
    width: 100%;
    margin: 0.25rem 0 0.45rem;
}

[class*="st-key-user_attachment_gallery_"] [data-testid="stImageContainer"] {
    margin-left: 0 !important;
    margin-right: 0.45rem !important;
}

[data-testid="stChatInput"] [data-testid="stFileChips"] {
    width: 100% !important;
    display: flex !important;
    flex-flow: row wrap !important;
    justify-content: flex-start !important;
    align-content: flex-start !important;
    align-self: stretch !important;
    align-items: flex-start !important;
    direction: ltr !important;
}

[data-testid="stChatInput"] [data-testid="stFileChips"] > div {
    flex: 0 0 auto !important;
    margin: 0 !important;
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


STORAGE_INIT_KEY = "storage_initialized"
DATABASE_PATH_KEY = "storage_database_path"
CONVERSATION_CACHE_KEY = "conversation_cache"
JOB_POLL_KEY_PREFIX = "polling_generation_job_"
MEMORY_POLL_KEY_PREFIX = "polling_memory_analysis_"


def _shutdown_generation_task_manager(manager: GenerationTaskManager) -> None:
    manager.shutdown()


@st.cache_resource(
    show_spinner=False,
    on_release=_shutdown_generation_task_manager,
)
def get_generation_task_manager(db_path: str) -> GenerationTaskManager:
    """按数据库路径创建进程级 Manager，并执行一次幂等启动恢复。"""
    manager = GenerationTaskManager(db_path=db_path)
    manager.recover()
    return manager


def _shutdown_memory_analysis_task_manager(
    manager: MemoryAnalysisTaskManager,
) -> None:
    manager.shutdown()


@st.cache_resource(
    show_spinner=False,
    on_release=_shutdown_memory_analysis_task_manager,
)
def get_memory_analysis_task_manager() -> MemoryAnalysisTaskManager:
    """创建不占用 Generation Worker 的独立单线程管理器。"""
    return MemoryAnalysisTaskManager()


def install_scroll_position_guard(conversation_id: str | None) -> None:
    """仅在整页 rerun 时恢复历史浏览位置；底部用户仍自然跟随新回答。"""
    safe_id = "draft" if not conversation_id else "".join(
        char for char in conversation_id if char.isalnum() or char in "-_"
    )
    st.html(
        f"""
<script>
(() => {{
  const storageKey = "physics-agent-scroll-{safe_id}";
  const scroller = document.querySelector('[data-testid="stAppViewContainer"]')
    || document.scrollingElement;
  if (!scroller) return;
  if (scroller.__physicsScrollHandler) {{
    scroller.removeEventListener("scroll", scroller.__physicsScrollHandler);
  }}
  scroller.dataset.physicsScrollGuard = storageKey;
  const saved = JSON.parse(sessionStorage.getItem(storageKey) || "null");
  if (saved && saved.nearBottom === false) {{
    requestAnimationFrame(() => requestAnimationFrame(() => {{
      scroller.scrollTop = saved.top;
    }}));
  }}
  const remember = () => {{
    const distance = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    sessionStorage.setItem(storageKey, JSON.stringify({{
      top: scroller.scrollTop,
      nearBottom: distance < 120
    }}));
  }};
  scroller.__physicsScrollHandler = remember;
  scroller.addEventListener("scroll", remember, {{passive: true}});
  remember();
}})();
</script>
""",
        unsafe_allow_javascript=True,
    )


def clear_pending_image_state() -> None:
    """清除未发送图片、待确认状态和粘贴桥，不影响 SQLite 中的消息。"""
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


def initialize_storage() -> bool:
    """初始化 SQLite；失败时显示安全错误，不泄露路径、SQL 或密钥。"""
    try:
        db_path = initialize_database()
        get_generation_task_manager(db_path)
        st.session_state[DATABASE_PATH_KEY] = db_path
        st.session_state[STORAGE_INIT_KEY] = True
        return True
    except Exception:
        st.error("本地存储初始化失败，请稍后重试。")
        return False


def refresh_conversation_cache() -> None:
    st.session_state[CONVERSATION_CACHE_KEY] = list_conversations()


def get_conversation_cache() -> list[dict[str, object]]:
    return st.session_state.get(CONVERSATION_CACHE_KEY) or []


def ensure_current_conversation() -> str | None:
    """保留有效历史会话；其余情况进入不落库的临时新对话。"""
    current = st.session_state.get("current_conversation_id")
    if current is not None and get_conversation(current) is not None:
        return current
    st.session_state.current_conversation_id = None
    return None


def select_conversation(conversation_id: str) -> None:
    """切换会话并清理跨会话的临时图片状态。"""
    if conversation_id == st.session_state.get("current_conversation_id"):
        return
    _purge_memory_candidates_for_conversation(
        st.session_state.get("current_conversation_id")
    )
    st.session_state.current_conversation_id = conversation_id
    st.session_state.pop("conversation_action_error", None)
    clear_pending_image_state()
    st.rerun()


def create_new_conversation() -> None:
    _purge_memory_candidates_for_conversation(
        st.session_state.get("current_conversation_id")
    )
    st.session_state.current_conversation_id = None
    st.session_state.pop("conversation_action_error", None)
    clear_pending_image_state()
    st.rerun()


def rename_conversation_from_sidebar(conversation_id: str, title: str) -> None:
    title = str(title).strip()
    if title:
        rename_conversation(conversation_id, title)
        st.session_state.pop("conversation_action_error", None)
        refresh_conversation_cache()
        st.rerun()


def _conversation_has_active_job(conversation_id: str) -> bool:
    return bool(
        conversation_id
        and get_active_generation_job(conversation_id) is not None
    )


def delete_conversation_from_sidebar(conversation_id: str) -> None:
    """删除会话；若删除的是当前会话，自动选择剩余最近会话或新建。"""
    if _conversation_has_active_job(conversation_id):
        st.session_state.conversation_action_error = (
            "该会话正在生成回答，暂时不能删除。"
        )
        st.rerun()
    st.session_state.pop("conversation_action_error", None)
    _purge_memory_candidates_for_conversation(conversation_id)
    delete_conversation(conversation_id)
    delete_conversation_attachments(conversation_id)
    if conversation_id == st.session_state.get("current_conversation_id"):
        st.session_state.current_conversation_id = None
    clear_pending_image_state()
    refresh_conversation_cache()
    st.rerun()


def clear_current_conversation() -> None:
    """清空当前会话的消息、运行与状态，但保留会话本身。"""
    conversation_id = st.session_state.get("current_conversation_id")
    if not conversation_id:
        clear_pending_image_state()
        return
    if _conversation_has_active_job(conversation_id):
        st.session_state.conversation_action_error = (
            "当前会话正在生成回答，暂时不能清空。"
        )
        return
    st.session_state.pop("conversation_action_error", None)
    _purge_memory_candidates_for_conversation(
        st.session_state.current_conversation_id
    )
    clear_conversation_contents(conversation_id)
    delete_conversation_attachments(conversation_id)
    clear_pending_image_state()
    refresh_conversation_cache()
    st.rerun()


def _memory_candidate_key(
    conversation_id: str,
    assistant_message_id: str,
) -> str:
    return f"{conversation_id}:{assistant_message_id}"


def _purge_memory_candidates_for_conversation(conversation_id) -> None:
    """清理属于某会话的临时候选，避免泄漏到其他会话。"""
    candidates = st.session_state.get("memory_candidates")
    if not candidates or not conversation_id:
        return
    prefix = f"{conversation_id}:"
    for key in [key for key in candidates if str(key).startswith(prefix)]:
        candidates.pop(key, None)


def _safe_memory_history(conversation_id: str) -> list[dict[str, str]]:
    """最近完整轮次（去掉本轮），只含 role/content，无图片与工具内部数据。"""
    messages = list_messages(conversation_id)
    prior = messages[:-2] if len(messages) >= 2 else []
    stored = [StoredMessage.model_validate(item) for item in prior]
    return build_recent_history(stored, max_turns=3)


def _run_memory_analysis(
    manager: MemoryAnalysisTaskManager,
    candidate_key: str,
    conversation_id: str,
    user_message_id: str,
    user_model_content: str,
    assistant_display_content: str,
) -> None:
    """仅收集安全输入并提交独立后台任务。"""
    history = _safe_memory_history(conversation_id)
    try:
        manager.submit(
            conversation_id,
            candidate_key.split(":", 1)[1],
            user_message_id=user_message_id,
            user_question=user_model_content,
            assistant_answer=assistant_display_content,
            conversation_history=history,
        )
    except MemoryAnalysisTaskError:
        st.session_state.setdefault("memory_candidates", {})[candidate_key] = {
            "conversation_id": conversation_id,
            "user_message_id": user_message_id,
            "status": "error",
            "error": "记忆分析失败，请稍后重试。",
            "candidates": [],
            "statuses": [],
        }
        return


def _sync_memory_task_entry(
    manager: MemoryAnalysisTaskManager,
    candidate_key: str,
    conversation_id: str,
    assistant_message_id: str,
    user_message_id: str,
) -> dict[str, object] | None:
    """把进程级任务的安全结果投影到当前页面会话。"""
    task = manager.get(conversation_id, assistant_message_id)
    if task is None:
        return st.session_state.get("memory_candidates", {}).get(candidate_key)
    status = str(task.get("status", ""))
    if status in {"pending", "running"}:
        return task
    existing = st.session_state.setdefault("memory_candidates", {}).get(
        candidate_key
    )
    if existing is not None:
        return existing
    candidates = task.get("candidates") if status == "completed" else []
    entry = {
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "status": "done" if status == "completed" else "error",
        "error": task.get("error"),
        "candidates": candidates if isinstance(candidates, list) else [],
        "statuses": ["pending"] * len(candidates or []),
    }
    st.session_state["memory_candidates"][candidate_key] = entry
    return entry


def _render_memory_candidate_entry(
    candidate_key: str,
    assistant_message_id: str,
    entry: dict[str, object],
) -> None:
    if entry.get("error"):
        st.error(str(entry["error"]))
        return
    candidates = entry.get("candidates") or []
    statuses = entry.get("statuses") or []
    if not candidates:
        st.caption("本轮没有提取到可确认的长期记忆。")
        return
    for index, candidate in enumerate(candidates):
        memory_type = str(candidate.get("memory_type", ""))
        label = MEMORY_TYPE_LABELS.get(memory_type, memory_type)
        status = statuses[index] if index < len(statuses) else "pending"
        st.markdown(
            f"**{label}** · {candidate.get('topic', '')}  \n"
            f"{candidate.get('content', '')}  \n"
            f"置信度：{candidate.get('confidence', 0)} · "
            f"依据：{candidate.get('evidence_summary', '')}"
        )
        if status == "pending":
            confirm_col, ignore_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "确认保存",
                    key=f"confirm_memory_{assistant_message_id}_{index}",
                    width="stretch",
                ):
                    _confirm_memory_candidate(candidate_key, index)
                    st.rerun()
            with ignore_col:
                if st.button(
                    "忽略",
                    key=f"ignore_memory_{assistant_message_id}_{index}",
                    width="stretch",
                ):
                    _ignore_memory_candidate(candidate_key, index)
                    st.rerun()
        elif status == "confirmed":
            st.caption("已保存")
        elif status == "ignored":
            st.caption("已忽略")


@st.fragment(run_every=1.0)
def poll_memory_analysis(
    manager: MemoryAnalysisTaskManager,
    candidate_key: str,
    conversation_id: str,
    assistant_message_id: str,
    user_message_id: str,
) -> None:
    """只局部读取记忆分析状态，不阻塞聊天与 Generation Worker。"""
    entry = _sync_memory_task_entry(
        manager,
        candidate_key,
        conversation_id,
        assistant_message_id,
        user_message_id,
    )
    if entry and entry.get("status") in {"pending", "running"}:
        st.info("正在分析学习表现…")
        return
    if entry:
        _render_memory_candidate_entry(
            candidate_key,
            assistant_message_id,
            entry,
        )


def _confirm_memory_candidate(candidate_key: str, index: int) -> None:
    """确认保存候选；确认后才写入 learning_memories，重复确认走去重合并。"""
    entry = st.session_state.get("memory_candidates", {}).get(candidate_key)
    if not entry or entry.get("status") != "done":
        return
    candidates = entry.get("candidates") or []
    if index >= len(candidates):
        return
    try:
        candidate = MemoryCandidate.model_validate(dict(candidates[index]))
        confirm_memory_candidate(
            candidate,
            source_conversation_id=entry.get("conversation_id"),
            source_message_id=entry.get("user_message_id"),
        )
    except MemoryServiceError:
        entry["error"] = "记忆保存失败，请稍后重试。"
        return
    statuses = list(entry.get("statuses") or [])
    while len(statuses) <= index:
        statuses.append("pending")
    statuses[index] = "confirmed"
    entry["statuses"] = statuses


def _ignore_memory_candidate(candidate_key: str, index: int) -> None:
    """忽略候选：只删除当前临时候选，不写数据库。"""
    entry = st.session_state.get("memory_candidates", {}).get(candidate_key)
    if not entry:
        return
    statuses = list(entry.get("statuses") or [])
    while len(statuses) <= index:
        statuses.append("pending")
    statuses[index] = "ignored"
    entry["statuses"] = statuses


def _assistant_can_analyze(
    run: dict[str, object],
    content: str,
) -> bool:
    """只有存在对应 user、assistant 与 completed agent_run 时才可分析。"""
    if not run:
        return False
    if str(run.get("status", "")) not in ("completed", "completed_with_fallback"):
        return False
    tool_records = run.get("tool_records_json") or []
    if any(
        isinstance(record, dict) and record.get("status") == "error"
        for record in tool_records
    ):
        return False
    return not content.startswith(MEMORY_SYSTEM_ERROR_PREFIXES)


def render_memory_analysis(message: dict[str, object]) -> None:
    """在成功的 assistant 回答下方提供“分析本轮学习表现”。"""
    conversation_id = message.get("_conversation_id")
    assistant_message_id = message.get("_assistant_message_id")
    if not conversation_id or not assistant_message_id:
        return
    if not message.get("_can_analyze"):
        return
    user_model_content = message.get("_user_model_content")
    user_message_id = message.get("_user_message_id")
    if not user_model_content or not user_message_id:
        return

    candidate_key = _memory_candidate_key(conversation_id, assistant_message_id)
    manager = get_memory_analysis_task_manager()
    entry = _sync_memory_task_entry(
        manager,
        candidate_key,
        str(conversation_id),
        str(assistant_message_id),
        str(user_message_id),
    )
    if entry is None:
        if st.button(
            "分析本轮学习表现",
            key=f"analyze_memory_{assistant_message_id}",
        ):
            _run_memory_analysis(
                manager,
                candidate_key,
                conversation_id,
                user_message_id,
                user_model_content,
                str(message.get("content", "")),
            )
            st.rerun()
        return
    if not entry:
        return
    if entry.get("status") in {"pending", "running"}:
        poll_memory_analysis(
            manager,
            candidate_key,
            str(conversation_id),
            str(assistant_message_id),
            str(user_message_id),
        )
        return
    _render_memory_candidate_entry(
        candidate_key,
        str(assistant_message_id),
        entry,
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
        relation = batch.get("relation")
        if isinstance(relation, BatchRelationAnalysis):
            st.markdown(
                f"**多图关系：** `{relation.relationship.value}`  \n"
                f"**判断：** {relation.short_reason}"
            )
            if relation.image_roles:
                st.caption(
                    " · ".join(
                        f"图片{item.index}={item.role}"
                        for item in relation.image_roles
                    )
                )
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
    if not isinstance(analysis, dict) and not isinstance(route, dict):
        return
    if not isinstance(analysis, dict):
        analysis = {}
    if not isinstance(route, dict):
        route = {}

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
    message_id = str(message.get("_message_id", ""))
    safe_anchor = message_anchor(message_id)
    if safe_anchor:
        st.html(f'<span id="{safe_anchor}"></span>')
    if is_user:
        attachment_sources: list[object] = []
        attachment_captions: list[str] = []
        for attachment in message.get("attachments", []) or []:
            if not isinstance(attachment, dict):
                continue
            path = resolve_attachment_path(attachment)
            if path is not None:
                attachment_sources.append(str(path))
                attachment_captions.append(str(attachment.get("filename", "")))
        for preview in message.get("_preview_images", []) or []:
            if isinstance(preview, dict) and isinstance(preview.get("bytes"), bytes):
                attachment_sources.append(preview["bytes"])
                attachment_captions.append(str(preview.get("filename", "")))
        if attachment_sources:
            with st.container(
                key=f"user_attachment_gallery_{message_index}",
                horizontal_alignment="left",
            ):
                st.image(
                    attachment_sources,
                    caption=attachment_captions,
                    width=150,
                )
    alignment = "right" if is_user else "left"
    row_key = f"{role}_message_row_{message_index}"
    bubble_key = f"{role}_message_bubble_{message_index}"

    content = str(message.get("content", ""))
    if is_user and not content.strip():
        return
    with st.container(key=row_key, horizontal_alignment=alignment):
        with st.container(key=bubble_key, width="stretch"):
            st.markdown(content)
            if not is_user:
                render_rag_sources(message)
                render_agent_decision(message)
                render_tool_records(message)
                render_run_trace(message)
                render_memory_analysis(message)


def _tool_model_requests_from_trace(
    trace: dict[str, object],
    *,
    legacy_value: int = 0,
) -> int:
    """汇总 Tool Client 请求；无步骤的旧记录保留其已存数量。"""
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return 0
    total = 0
    found_tool_step = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("name") not in {"tool_selection", "tool_result_answer"}:
            continue
        found_tool_step = True
        requests = step.get("model_requests", 0)
        if isinstance(requests, int) and not isinstance(requests, bool):
            total += max(requests, 0)
    return total if found_tool_step else max(legacy_value, 0)


def _assistant_render_dict(
    message: dict[str, object],
    run_by_assistant_id: dict[str, dict[str, object]],
    *,
    user_context: dict[str, object] | None = None,
    can_analyze: bool = False,
) -> dict[str, object]:
    """从消息与对应 agent_run 还原 assistant 的页面展示详情。"""
    run = run_by_assistant_id.get(str(message.get("id", ""))) or {}
    trace = run.get("trace_json")
    if not isinstance(trace, dict):
        trace = {}
    analysis = run.get("analysis_json")
    if not isinstance(analysis, dict):
        analysis = None
    display_trace = trace if trace else None
    return {
        "role": "assistant",
        "content": message.get("display_content") or "",
        "sources": run.get("sources_json") or [],
        "analysis": analysis,
        "route": {
            "teaching_mode": run.get("teaching_mode"),
            "use_rag": bool(run.get("use_rag")),
            "use_tools": bool(run.get("use_tools")),
            "should_answer": trace.get("status") != "blocked",
        },
        "analysis_fallback": bool(trace.get("analysis_fallback", False)),
        "tool_records": run.get("tool_records_json") or [],
        "tool_model_requests": _tool_model_requests_from_trace(
            trace,
            legacy_value=(
                int(run.get("total_model_requests", 0) or 0)
                if run.get("tool_records_json")
                else 0
            ),
        ),
        "trace": display_trace,
        "_conversation_id": message.get("conversation_id"),
        "_assistant_message_id": message.get("id"),
        "_user_message_id": (
            user_context.get("id") if user_context else None
        ),
        "_user_model_content": (
            user_context.get("model_content") if user_context else None
        ),
        "_can_analyze": can_analyze,
    }


def _user_render_dict(message: dict[str, object]) -> dict[str, object]:
    metadata = message.get("image_metadata_json") or []
    metadata_items = metadata if isinstance(metadata, list) else [metadata]
    attachments: list[dict[str, object]] = []
    for item in metadata_items:
        if not isinstance(item, dict):
            continue
        stored = item.get("attachments")
        if isinstance(stored, list):
            attachments.extend(
                attachment for attachment in stored if isinstance(attachment, dict)
            )
    return {
        "role": "user",
        "content": message.get("display_content") or "",
        "_message_id": message.get("id"),
        "attachments": attachments,
    }


def load_render_messages(
    conversation_id: str,
) -> list[dict[str, object]]:
    """从 SQLite 读取消息，并用 agent_run 的 JSON 字段还原 assistant 详情。"""
    messages = list_messages(conversation_id)
    runs = get_agent_runs(conversation_id)
    run_by_assistant_id = {
        str(run["assistant_message_id"]): run
        for run in runs
        if run.get("assistant_message_id")
    }
    rendered: list[dict[str, object]] = []
    last_user: dict[str, object] | None = None
    for message in messages:
        if message.get("role") == "assistant":
            run = run_by_assistant_id.get(str(message.get("id", ""))) or {}
            can_analyze = _assistant_can_analyze(
                run,
                str(message.get("display_content", "")),
            )
            rendered.append(
                _assistant_render_dict(
                    message,
                    run_by_assistant_id,
                    user_context=last_user,
                    can_analyze=can_analyze,
                )
            )
        else:
            last_user = message
            rendered.append(_user_render_dict(message))
    return rendered


def _latest_generation_job(conversation_id: str) -> dict[str, object] | None:
    jobs = list_generation_jobs(conversation_id)
    return jobs[-1] if jobs else None


def _retry_generation_job_from_page(
    job_id: str,
    manager: GenerationTaskManager,
) -> None:
    try:
        manager.retry(job_id)
    except TaskManagerError:
        st.session_state.last_turn_error = "该回答任务暂时无法重新生成。"
    else:
        st.session_state.pop("last_turn_error", None)
    st.rerun()


def render_generation_job_status(
    job: dict[str, object] | None,
    manager: GenerationTaskManager,
) -> None:
    """展示安全 Job 状态；completed 不显示持续状态或 Retry。"""
    if not job:
        return
    status = str(job.get("status", ""))
    if status == "pending":
        st.info("等待生成")
    elif status == "running":
        st.info("正在生成")
    elif status == "failed":
        st.error("本轮回答生成失败，可以重新生成。")
        if st.button("重新生成", key=f"retry_job_{job.get('id', '')}"):
            _retry_generation_job_from_page(str(job.get("id", "")), manager)
    elif status == "interrupted":
        st.warning("上次生成被服务中断，可以重新生成。")
        if st.button("重新生成", key=f"retry_job_{job.get('id', '')}"):
            _retry_generation_job_from_page(str(job.get("id", "")), manager)


@st.fragment(run_every=1.0)
def poll_active_generation_job(
    conversation_id: str,
    manager: GenerationTaskManager,
) -> None:
    """只读轮询当前会话；终态出现时触发一次完整页面刷新。"""
    polling_key = f"{JOB_POLL_KEY_PREFIX}{conversation_id}"
    active_job = get_active_generation_job(conversation_id)
    if active_job is None:
        if st.session_state.pop(polling_key, None) is not None:
            st.rerun()
        return
    st.session_state[polling_key] = active_job["id"]
    render_generation_job_status(active_job, manager)


def _image_history_metadata(
    batch: dict[str, object] | None,
    *,
    context_used: bool,
    attachments: list[dict[str, object]] | None = None,
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
            "attachments": attachments or [],
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
        "attachments": attachments or [],
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
    display_question: str,
    model_question: str,
    batch: dict[str, object],
    original_images: list[dict[str, object]],
) -> None:
    """暂存确认所需数据；原图 bytes 只存在本次待发送会话内。"""
    st.session_state.pending_submission = {
        "display_question": display_question,
        "model_question": model_question,
        "batch": batch,
        "original_images": original_images,
    }
    batch_id = str(batch.get("batch_id", ""))
    images = batch.get("images", [])
    if not isinstance(images, list):
        return
    relation = batch.get("relation")
    relation_uncertain = bool(
        isinstance(relation, BatchRelationAnalysis)
        and relation.relationship is BatchRelation.UNCERTAIN
    )
    for record in images:
        if not isinstance(record, dict) or not (
            relation_uncertain or image_needs_confirmation(record)
        ):
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
    display_question = pending.get("display_question")
    model_question = pending.get("model_question")
    batch = pending.get("batch")
    original_images = pending.get("original_images")
    if (
        not isinstance(display_question, str)
        or not isinstance(model_question, str)
        or not isinstance(batch, dict)
        or not isinstance(original_images, list)
    ):
        return None
    images = batch.get("images", [])
    if not isinstance(images, list):
        return None

    st.warning("部分题图需要你核对。确认前不会调用教师 Agent。")
    render_batch_details(batch)
    batch_id = str(batch.get("batch_id", ""))
    relation = batch.get("relation")
    relation_uncertain = bool(
        isinstance(relation, BatchRelationAnalysis)
        and relation.relationship is BatchRelation.UNCERTAIN
    )
    confirmation_records = [
        record
        for record in images
        if isinstance(record, dict)
        and (relation_uncertain or image_needs_confirmation(record))
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
            "display_question": display_question,
            "model_question": model_question,
            "batch": batch,
            "original_images": original_images,
            "image_context": build_batch_image_context(
                images,
                context_overrides=overrides,
                relation=relation if isinstance(relation, BatchRelationAnalysis) else None,
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
st.session_state.setdefault("vision_batch_cache", {})
st.session_state.setdefault("paste_bridge_reset_token", 0)
st.session_state.setdefault("memory_candidates", {})

if not st.session_state.get(STORAGE_INIT_KEY):
    if not initialize_storage():
        st.stop()

database_path = st.session_state.get(DATABASE_PATH_KEY)
if not isinstance(database_path, str) or not database_path:
    if not initialize_storage():
        st.stop()
    database_path = st.session_state[DATABASE_PATH_KEY]
task_manager = get_generation_task_manager(database_path)

current_conversation_id = ensure_current_conversation()
refresh_conversation_cache()
install_scroll_position_guard(current_conversation_id)
render_messages = (
    load_render_messages(current_conversation_id)
    if current_conversation_id
    else []
)

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
            submission_to_process = {
                "display_question": submitted_text,
                "model_question": submitted_text or DEFAULT_IMAGE_QUESTION,
                "images": queued_images,
            }

with st.sidebar:
    st.markdown("**初中物理教师**")
    st.caption(
        "面向初中物理学习的个人教师 Agent，支持普通问答、"
        "本地知识库辅助与本地确定性计算工具。"
    )
    st.button(
        "清空当前会话",
        key="clear_conversation",
        icon=":material/edit_square:",
        width="stretch",
        on_click=clear_current_conversation,
    )
    if st.session_state.get("conversation_action_error"):
        st.warning(st.session_state["conversation_action_error"])

    st.markdown("##### 会话")
    if st.button(
        "新建对话",
        key="new_conversation",
        icon=":material/add:",
        width="stretch",
    ):
        create_new_conversation()
    for conversation in get_conversation_cache():
        cid = str(conversation.get("id", ""))
        is_current = cid == current_conversation_id
        with st.container(key=f"conversation_item_{cid}"):
            updated_title = editable_conversation_title(
                str(conversation.get("title", "新对话")),
                current=is_current,
                key=f"conversation_title_{cid}",
            )
            if updated_title is not None:
                rename_conversation_from_sidebar(cid, updated_title)
            select_col, delete_col = st.columns(2)
            with select_col:
                if st.button("打开", key=f"select_conv_{cid}", width="content"):
                    select_conversation(cid)
            with delete_col:
                if st.button("删除", key=f"delete_conv_{cid}", width="content"):
                    delete_conversation_from_sidebar(cid)
    st.divider()

    st.markdown("##### 当前对话")
    user_message_previews = [
        (
            str(message.get("_message_id", "")),
            str(message.get("content", "")).strip()
            or "[图片]",
        )
        for message in render_messages
        if message.get("role") == "user"
    ]
    if user_message_previews:
        with st.container(key="current_conversation_outline"):
            for preview_index, (message_id, preview) in enumerate(
                user_message_previews
            ):
                preview_text = preview if len(preview) <= 28 else f"{preview[:28]}…"
                anchor_id = message_anchor(message_id)
                with st.container(key=f"conversation_preview_{preview_index}"):
                    st.markdown(
                        f"[:material/chat_bubble_outline: {preview_text}]"
                        f"(#{anchor_id})"
                    )
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

    st.divider()
    st.markdown("##### 学习档案")
    memories = [
        memory
        for memory in list_memories()
        if memory.get("confirmed") == 1
    ]
    if memories:
        for memory in memories:
            memory_id = str(memory.get("id", ""))
            memory_type = str(memory.get("memory_type", ""))
            label = MEMORY_TYPE_LABELS.get(memory_type, memory_type)
            with st.container(key=f"memory_item_{memory_id}"):
                st.markdown(
                    f"**{label}** · {memory.get('topic', '')}  \n"
                    f"{memory.get('content', '')}  \n"
                    f"证据次数：{memory.get('evidence_count', 0)}"
                )
                action_col, delete_col = st.columns(2)
                with action_col:
                    if st.button(
                        "停用",
                        key=f"memory_deactivate_{memory_id}",
                        width="content",
                    ):
                        deactivate_memory(memory_id)
                        st.rerun()
                with delete_col:
                    if st.button(
                        "删除",
                        key=f"memory_delete_{memory_id}",
                        width="content",
                    ):
                        delete_memory(memory_id)
                        st.rerun()
    else:
        st.caption("还没有已确认的长期记忆。")

is_empty_state = (
    not render_messages
    and submission_to_process is None
    and st.session_state.get("pending_submission") is None
)
response_slot = None
provisional_user_rendered = False

if is_empty_state:
    with st.container(key="empty_state"):
        st.markdown("## 今天想从哪道物理题开始？")
        if submission_error:
            st.warning(submission_error)
else:
    for message_index, message in enumerate(render_messages):
        render_chat_message(message, message_index)

    if submission_to_process is not None:
        render_chat_message(
            {
                "role": "user",
                "content": str(submission_to_process["display_question"]),
                "_preview_images": submission_to_process["images"],
            },
            len(render_messages),
        )
        provisional_user_rendered = True
    response_slot = st.empty()

active_generation_job = (
    get_active_generation_job(current_conversation_id)
    if current_conversation_id
    else None
)
if active_generation_job is not None:
    poll_active_generation_job(current_conversation_id, task_manager)
else:
    st.session_state.pop(
        f"{JOB_POLL_KEY_PREFIX}{current_conversation_id}",
        None,
    )
    if current_conversation_id:
        render_generation_job_status(
            _latest_generation_job(current_conversation_id),
            task_manager,
        )

if st.session_state.get("last_turn_error"):
    st.error(st.session_state["last_turn_error"])

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
    disabled=(
        st.session_state.get("pending_submission") is not None
        or active_generation_job is not None
    ),
    on_submit=queue_chat_submission,
    args=("question_input",),
)

agent_payload = pending_agent_payload
if submission_error and response_slot is not None:
    with response_slot.container():
        st.warning(submission_error)

if submission_to_process is not None:
    display_question = str(submission_to_process["display_question"])
    model_question = str(submission_to_process["model_question"])
    images = submission_to_process["images"]
    if images:
        try:
            with response_slot.container():
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
            elif batch_needs_confirmation(batch):
                _store_pending_submission(
                    display_question,
                    model_question,
                    batch,
                    images,
                )
                if response_slot is not None:
                    with response_slot.container():
                        render_pending_confirmation()
            else:
                with response_slot.container():
                    render_batch_details(batch)
                agent_payload = {
                    "display_question": display_question,
                    "model_question": model_question,
                    "batch": batch,
                    "original_images": images,
                    "image_context": build_batch_image_context(
                        batch_images,
                        relation=(
                            batch.get("relation")
                            if isinstance(
                                batch.get("relation"),
                                BatchRelationAnalysis,
                            )
                            else None
                        ),
                    ),
                }
    else:
        agent_payload = {
            "display_question": display_question,
            "model_question": model_question,
            "batch": None,
            "original_images": [],
            "image_context": None,
        }

if agent_payload is not None:
    display_question = str(agent_payload["display_question"])
    model_question = str(agent_payload["model_question"])
    original_images = agent_payload.get("original_images") or []
    image_context = agent_payload.get("image_context")
    st.session_state.pop("last_turn_error", None)
    created_draft_conversation = False
    stored_attachments: list[dict[str, object]] = []
    turn_enqueued = False
    try:
        with response_slot.container():
            if not provisional_user_rendered:
                render_chat_message(
                    {
                        "role": "user",
                        "content": display_question,
                        "_preview_images": original_images,
                    },
                    len(render_messages),
                )
            with st.status("正在提交生成任务……", expanded=False) as answer_status:
                if not current_conversation_id:
                    title_seed = display_question or "图片题"
                    conversation = create_conversation(title_seed[:30])
                    current_conversation_id = str(conversation["id"])
                    st.session_state.current_conversation_id = current_conversation_id
                    created_draft_conversation = True
                    refresh_conversation_cache()
                if original_images:
                    stored_attachments = save_image_attachments(
                        current_conversation_id,
                        original_images,
                    )
                image_metadata = _image_history_metadata(
                    agent_payload.get("batch"),
                    context_used=bool(image_context),
                    attachments=stored_attachments,
                ) if original_images else None
                enqueued = enqueue_conversation_turn(
                    current_conversation_id,
                    display_question=display_question,
                    model_question=model_question,
                    image_metadata=image_metadata,
                    confirmed_image_context=(
                        image_context
                        if isinstance(image_context, str) and image_context.strip()
                        else None
                    ),
                    mode_override=mode_override_for_question,
                    rag_policy=rag_policy_for_question,
                )
                turn_enqueued = True
                task_manager.submit(enqueued["generation_job_id"])
                answer_status.update(label="已加入生成队列", state="complete")
    except (AttachmentStoreError, ConversationServiceError, TaskManagerError):
        if stored_attachments and not turn_enqueued:
            delete_saved_attachments(stored_attachments)
        if (
            created_draft_conversation
            and current_conversation_id
            and not turn_enqueued
        ):
            delete_conversation(current_conversation_id)
            st.session_state.current_conversation_id = None
        st.session_state.last_turn_error = "回答任务提交失败，请稍后重试。"
        st.rerun()
    else:
        _clear_pending_submission()
        st.rerun()
