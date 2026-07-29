# 当前状态

## Stage 03 状态

“初中物理教师 Agent + 阿里云百炼千问 API”已完成 teacher_v1 基础提示词接入和工程
回归，当前使用 `qwen3.7-flash`。teacher_v1 是临时基础版本，不是最终个人教学风格。

## 当前结构

- `app.py`：Streamlit 页面入口，负责聊天输入、加载提示、消息显示和当前页面历史
- `main.py`：固定平均速度题的终端回归入口
- `src/config.py`：使用 `python-dotenv` 加载并校验千问配置
- `src/prompts.py`：定义 `PROMPT_VERSION = "teacher_v1"` 和教师 system prompt
- `src/model_client.py`：从提示词模块导入 system prompt，并使用 OpenAI 兼容客户端调用千问
- `evaluation/stage03_prompt_cases.md`：记录固定行为测试题、测试目的和空白评分维度

API Key 仅保存在本地 `.env` 中；`.env` 和 `.venv` 均被 Git 忽略。

## 运行命令

```powershell
# 终端回归
.venv\Scripts\python.exe main.py

# 网页启动
.venv\Scripts\python.exe -m streamlit run app.py
```

## 会话行为

`st.session_state` 只保存当前浏览器会话/标签页中的消息，页面关闭或服务重启后不会持久化。
当前页面可以显示历史消息，但每次模型请求只发送当前问题，不发送完整页面历史。

## 验证状态

自动测试已确认：

- `qwen3.7-flash` 终端回答包含 `5 m/s`
- 空问题被直接拦截，未加载配置或调用模型
- 临时缺少 `QWEN_MODEL` 时错误清楚，测试进程结束后正常配置仍为 `qwen3.7-flash`
- Streamlit 启动检查返回 HTTP 200
- `compileall`、`pip check`、`git diff --check` 通过
- teacher_v1 修改后的 5 次固定题调用均返回非空 content
- 问题 4 能先定位第一处错误，但随后仍继续完整计算
- 问题 5 能指出缺少质量且不擅自假设，但回答仍略显冗长

用户已手动验证：

- 网页计算题返回 `5 m/s`
- 网页概念题正确解释金属和木头导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

这 5 道题只覆盖简单计算、概念解释、提示请求、错误诊断和条件不足处理，不代表完整初中
物理能力。本阶段未使用有代表性的个人题库，尚未证明 teacher_v1 全面优于旧提示词。

模型每次只接收当前问题，没有多轮记忆。尚不支持图片、数据库、RAG、工具调用、MCP 和
真正的多轮上下文，也没有加入后续阶段功能。
