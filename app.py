"""初中物理教师 Agent 的最小 Streamlit 聊天页面。"""

import streamlit as st

from src.model_client import answer_question


MODEL_ERROR_MESSAGES = {
    "认证失败：请检查 DASHSCOPE_API_KEY 是否正确。",
    "网络连接失败：请检查网络或 QWEN_BASE_URL。",
    "请求过于频繁：请稍后再试。",
    "千问 API 请求失败，请稍后再试。",
    "调用千问时发生未知错误，请稍后再试。",
    "千问返回了空回答。",
}


st.set_page_config(page_title="初中物理教师 Agent")
st.title("初中物理教师 Agent")
st.caption("输入一道初中物理问题，教师 Agent 将使用千问给出简洁回答。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("请输入你的初中物理问题")

if question is not None:
    question = question.strip()
    if not question:
        st.warning("问题不能为空，请输入一道初中物理问题。")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("教师正在思考，请稍候……"):
                answer = answer_question(question)
        except Exception:
            st.error("回答生成失败，请稍后再试。")
        else:
            if answer in MODEL_ERROR_MESSAGES:
                st.error(answer)
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                with st.chat_message("assistant"):
                    st.markdown(answer)
