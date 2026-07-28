# 当前状态

## 第 2 阶段状态

“初中物理教师 Agent + 阿里云百炼千问 API”已完成最小 Streamlit 文本聊天网页和工程
拆分，当前使用 `qwen3.7-flash`。

## 当前结构

- `app.py`：Streamlit 页面入口，负责聊天输入、加载提示、消息显示和当前页面历史
- `main.py`：固定平均速度题的终端回归入口
- `src/config.py`：使用 `python-dotenv` 加载并校验千问配置
- `src/model_client.py`：使用 OpenAI 兼容客户端调用千问并返回回答

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

用户已手动验证：

- 网页计算题返回 `5 m/s`
- 网页概念题正确解释金属和木头导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

尚不支持图片、数据库、RAG、工具调用和真正的多轮上下文，也没有加入后续阶段功能。
