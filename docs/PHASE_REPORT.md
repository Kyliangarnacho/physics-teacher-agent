# 阶段报告

## 第 1 阶段

完成“初中物理教师 Agent + 阿里云百炼千问 API”的基础文本问答、配置校验、错误处理和
终端测试。

## 第 2 阶段：最小 Streamlit 聊天网页

### 完成内容

- 将配置读取拆分到 `src/config.py`
- 将千问调用拆分到 `src/model_client.py`
- 保留 `main.py` 作为固定平均速度题的终端回归入口
- 新增 `app.py`，提供最小 Streamlit 文本聊天页面
- 将模型切换为 `qwen3.7-flash`
- 使用 `st.session_state` 保存当前页面的用户消息和教师回答
- 更新虚拟环境依赖并记录 Streamlit

### 运行方式

```powershell
# 终端回归
.venv\Scripts\python.exe main.py

# 网页启动
.venv\Scripts\python.exe -m streamlit run app.py
```

网页默认地址为 `http://localhost:8501`。

### 自动测试结果

1. `qwen3.7-flash` 成功回答平均速度题，结果包含 `5 m/s`。
2. 空问题在配置加载和模型调用前被拦截。
3. 临时进程环境缺少 `QWEN_MODEL` 时，程序给出清楚的中文配置错误；测试结束后正常
   配置仍为 `qwen3.7-flash`。
4. Streamlit 服务启动成功并返回 HTTP 200，随后正常停止。
5. `compileall`、`pip check` 和 `git diff --check` 均通过。

### 用户手动验证结果

- 网页计算题返回 `5 m/s`
- 网页概念题正确解释金属和木头的导热差异
- 当前页面能够显示两轮聊天记录

### 会话边界

`st.session_state` 仅保存当前浏览器会话/标签页的临时历史。页面虽然显示多轮消息，但
每次模型请求只发送当前问题，不发送完整历史，因此尚未实现真正的多轮上下文。

### 当前限制

当前不支持图片、数据库、RAG、工具调用或真正的多轮上下文，也未加入其他后续阶段功能。
