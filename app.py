"""初中物理教师 Agent 的最小 Streamlit 聊天页面。"""

import streamlit as st

from src.agent import run_teacher_agent


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
}
</style>
"""


def clear_conversation() -> None:
    """清空当前浏览器会话中的聊天记录。"""
    st.session_state.messages = []


def queue_question(input_key: str) -> None:
    """将聊天输入暂存到下一次脚本运行中。"""
    st.session_state.pending_question = st.session_state.get(input_key)


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
            f"**简短判断理由：** {analysis.get('short_reason', '无')}  \n"
            f"**Analyzer 是否发生 fallback：** {fallback}"
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


st.set_page_config(
    page_title="初中物理教师 Agent",
    page_icon=":material/science:",
    layout="centered",
    initial_sidebar_state="auto",
)
st.html(APP_STYLES)
st.session_state.setdefault("messages", [])
queued_question = st.session_state.pop("pending_question", None)
question_to_answer = None
question_error = None
mode_override_for_question = str(
    st.session_state.get("teaching_mode", "auto")
)
rag_policy_for_question = str(
    st.session_state.get("rag_policy", "auto")
)

if queued_question is not None:
    normalized_question = str(queued_question).strip()
    if normalized_question:
        question_to_answer = normalized_question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": normalized_question,
            }
        )
    else:
        question_error = "问题不能为空，请输入一道初中物理问题。"

with st.sidebar:
    st.markdown("**初中物理教师**")
    st.caption("面向初中物理学习的个人教师 Agent，支持普通问答与本地知识库辅助。")
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

is_empty_state = (
    not st.session_state.messages
    and question_to_answer is None
)
response_slot = None

if is_empty_state:
    with st.container(key="empty_state"):
        st.markdown("## 今天想从哪道物理题开始？")
        if question_error:
            st.warning(question_error)
else:
    for message_index, message in enumerate(st.session_state.messages):
        render_chat_message(message, message_index)

    response_slot = st.empty()

st.chat_input(
    "把题目或困惑发过来"
    if is_empty_state
    else "继续问一道物理问题…",
    key="question_input",
    on_submit=queue_question,
    args=("question_input",),
)

if not is_empty_state and response_slot is not None:
    if question_error:
        with response_slot.container():
            st.warning(question_error)
    elif question_to_answer is not None:
        try:
            with response_slot.container():
                with st.spinner("教师正在思考，请稍候……"):
                    agent_result = run_teacher_agent(
                        question_to_answer,
                        mode_override=mode_override_for_question,
                        rag_policy=rag_policy_for_question,
                    )
                    answer = agent_result["answer"]
        except Exception:
            with response_slot.container():
                st.error("回答生成失败，请稍后再试。")
        else:
            if answer in MODEL_ERROR_MESSAGES:
                with response_slot.container():
                    st.error(answer)
            else:
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
                }
                st.session_state.messages.append(assistant_message)
                with response_slot.container():
                    render_chat_message(
                        assistant_message,
                        len(st.session_state.messages) - 1,
                    )
