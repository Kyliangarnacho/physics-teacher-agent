"""初中物理教师 Agent 的最小 Streamlit 聊天页面。"""

import streamlit as st

from src.model_client import answer_question
from src.rag import answer_with_rag


MODEL_ERROR_MESSAGES = {
    "认证失败：请检查 DASHSCOPE_API_KEY 是否正确。",
    "网络连接失败：请检查网络或 QWEN_BASE_URL。",
    "请求过于频繁：请稍后再试。",
    "千问 API 请求失败，请稍后再试。",
    "调用千问时发生未知错误，请稍后再试。",
    "千问返回了空回答。",
}


def render_rag_sources(message: dict[str, object]) -> None:
    """重新渲染一次 RAG 回答的来源或无来源提示。"""
    if not message.get("rag_enabled"):
        return

    sources = message.get("sources", [])
    if not isinstance(sources, list) or not sources:
        st.info("本地知识库未检索到相关资料，本次按普通问答处理。")
        return

    with st.expander("本地知识库参考来源", expanded=False):
        for source in sources:
            if not isinstance(source, dict):
                continue
            st.markdown(
                f"- **{source.get('id', '')}**  \n"
                f"  主题：{source.get('topic', '')}  \n"
                f"  来源：{source.get('source', '')}"
            )


st.set_page_config(page_title="初中物理教师 Agent")
st.title("初中物理教师 Agent")
st.caption("输入一道初中物理问题，教师 Agent 将使用千问给出简洁回答。")
use_local_knowledge_base = st.checkbox(
    "使用本地物理知识库",
    value=False,
    key="use_local_knowledge_base",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_rag_sources(message)

question = st.chat_input("请输入你的初中物理问题")

if question is not None:
    question = question.strip()
    if not question:
        st.warning("问题不能为空，请输入一道初中物理问题。")
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
                "rag_enabled": use_local_knowledge_base,
            }
        )
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("教师正在思考，请稍候……"):
                if use_local_knowledge_base:
                    rag_result = answer_with_rag(question, top_k=3)
                    answer = rag_result["answer"]
                    sources = rag_result["sources"]
                else:
                    answer = answer_question(question)
                    sources = []
        except Exception:
            st.error("回答生成失败，请稍后再试。")
        else:
            if answer in MODEL_ERROR_MESSAGES:
                st.error(answer)
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "rag_enabled": use_local_knowledge_base,
                    "sources": sources,
                }
                st.session_state.messages.append(assistant_message)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    render_rag_sources(assistant_message)
