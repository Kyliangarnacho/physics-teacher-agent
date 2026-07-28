# Physics Teacher Agent

## 项目定位

本项目是“初中物理教师 Agent + 阿里云百炼千问 API”。当前已完成第 2 阶段的
最小 Streamlit 文本聊天网页，使用 `qwen3.7-flash` 回答初中物理问题。

## 环境与配置

- Python 3.12 虚拟环境：`.venv`
- 主要依赖：`openai`、`python-dotenv`、`streamlit`
- 本地配置：`.env`
- 当前模型：`qwen3.7-flash`

`.env` 保存 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL` 和 `QWEN_MODEL`，已被 Git
忽略且不会提交。可复制 `.env.example` 后填写本地 API Key。

## 代码职责

- `app.py`：提供最小 Streamlit 聊天页面，显示并保存当前页面的消息
- `main.py`：运行固定平均速度题，作为终端回归入口
- `src/config.py`：加载并校验 `.env` 中的千问配置
- `src/model_client.py`：校验问题、调用千问并返回回答或简短中文错误

## 运行

终端回归：

```powershell
.venv\Scripts\python.exe main.py
```

启动网页：

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

启动后默认访问 `http://localhost:8501`。

## 会话与请求行为

网页使用 `st.session_state` 保存当前页面的用户消息和教师回答。历史仅属于当前浏览器
会话/标签页，关闭页面或重启服务后不会持久保存。

页面可以显示多轮聊天记录，但每次调用模型时只发送当前问题，不发送页面中的完整历史，
因此目前不是真正的多轮上下文对话。

## 已完成验证

自动验证：

- 使用 `qwen3.7-flash` 运行终端平均速度题，回答包含 `5 m/s`
- 空问题在加载配置和调用模型前被拦截
- 临时缺少 `QWEN_MODEL` 时给出清楚的中文错误，测试后恢复正常配置
- Streamlit 服务启动并在 `http://localhost:8501` 返回 HTTP 200
- `compileall`、`pip check` 和 `git diff --check` 通过

用户手动验证的网页结果：

- 计算题返回 `5 m/s`
- 概念题正确解释金属和木头的导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

当前仅支持文本问题，不支持图片、数据库、RAG、工具调用或真正的多轮上下文，也未加入
其他 Agent 框架或后续阶段功能。
