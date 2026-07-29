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

## Stage 03：teacher_v1 基础提示词

### 工程变更

- 新增 `src/prompts.py`，定义 `PROMPT_VERSION = "teacher_v1"` 和教师 system prompt
- `src/model_client.py` 改为从提示词模块导入 system prompt
- 保持模型、API 参数、回答提取接口和单问题调用方式不变
- 新增 `evaluation/stage03_prompt_cases.md`，记录 5 道固定行为测试题、目的和评分维度

teacher_v1 是用于建立提示词评测流程的临时基础版本，不是最终个人教学风格。

### 固定行为测试

5 道题主要验证简单计算、概念解释、提示请求、错误诊断和条件不足处理。teacher_v1 修改后
5 次调用均返回非空 content：

- 问题 1 得到 `5 m/s`
- 问题 3 只给提示，没有泄露最终数值
- 问题 4 先指出第一处错误，但随后仍继续完整计算
- 问题 5 指出缺少质量且没有擅自假设，但回答仍略显冗长

这些题不代表完整初中物理能力。本阶段没有使用有代表性的个人题库，不能据此证明
teacher_v1 全面优于旧提示词。

### 人工评分与参与验收

评分由项目发起人根据 5 道固定题的真实回答确认。项目发起人已完成人工评分和效果判断：

- teacher_v1 得分为 46/50，旧提示词基线为 47/50
- teacher_v1 的题型规则更清楚，提示题没有泄露答案，emoji 和客套内容减少
- 问题 4 仍在定位错误后继续完整计算
- 问题 2 和问题 5 仍有冗长
- 五题小样本不能证明 teacher_v1 全面优于旧提示词

项目发起人的参与验收已满足。

### 工程回归

- `.venv\Scripts\python.exe main.py`：成功，回答包含 `5 m/s`
- Streamlit：启动成功，`http://localhost:8501` 返回 HTTP 200 后停止测试进程
- `python -m compileall app.py main.py src`：通过
- `python -m pip check`：通过
- `git diff --check`：通过
- `git check-ignore -v .env`：确认 `.env` 被忽略

### 当前边界

模型仍只接收当前问题，没有多轮记忆。当前未实现图片、RAG、工具、数据库、MCP 或真正的
多轮上下文。
