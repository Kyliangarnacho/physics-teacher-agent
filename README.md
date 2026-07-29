# Physics Teacher Agent

## 项目定位

本项目是“初中物理教师 Agent + 阿里云百炼千问 API”。当前进入 Stage 04 工程收尾，
使用 `qwen3.7-flash` 回答初中物理问题，提示词版本冻结为
`teacher_v3_personal_humor`。

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
- `src/prompts.py`：保存提示词版本和初中物理教师 system prompt
- `src/model_client.py`：从提示词模块导入 system prompt，校验问题、调用千问并返回回答
- `evaluation/stage04_text_cases_v1.json`：保存 15 道 Stage 04 纯文本评测题
- `evaluation/results/`：保存不同提示词版本的 JSONL 评测结果
- `evaluation/reviews/`：保存人工评审表与版本对比
- `evaluation/validate_stage04_text_cases.py`：校验题目字段、数量和 ID
- `evaluation/run_stage04_text_evaluation.py`：支持 `--limit` 和断点续跑的评测入口
- `tests/`：使用 Python 标准库 `unittest` 验证评测工具

## Stage 04 提示词与评测状态

当前提示词版本为 `teacher_v3_personal_humor`，本阶段已冻结，不再依据现有 5 道题继续
调整。个人风格以讲解逻辑、条件分类、因果链和纠错方式为核心；幽默只在语境自然匹配时
偶尔出现，不要求每题都有。

已建立 15 道文本评测题。`teacher_v2_personal` 和
`teacher_v3_personal_humor` 均只完成了前 5 道真实调用，尚未运行全部 15 道。
S04-TXT-001 已由用户人工评分为 7/8，其余题目的人工评分仍待完成。

原始 PDF、Word 和私有风格示例仅保存在 Git 忽略的 `data/raw/` 与
`data/private_evaluation/`，不会提交 Git。可提交的重写评测题和元数据位于
`evaluation/`。

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

本地校验与单元测试：

```powershell
.venv\Scripts\python.exe evaluation\validate_stage04_text_cases.py
.venv\Scripts\python.exe -m unittest discover -v
```

运行前 5 道真实评测时必须显式使用独立结果文件，以保留不同提示词版本：

```powershell
.venv\Scripts\python.exe evaluation\run_stage04_text_evaluation.py `
  --limit 5 `
  --output evaluation\results\stage04_text_v1_teacher_v3_personal_humor_results.jsonl
```

评测脚本会跳过结果文件中已存在的 ID。该命令会产生 API 调用；仅在需要新评测且已确认
API 配置与费用时执行。

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
- Stage 04 评测集为 15 题，字段完整且 ID 唯一
- `teacher_v2_personal` 与 `teacher_v3_personal_humor` 的前 5 题结果均为非空
  `content`

用户手动验证的网页结果：

- 计算题返回 `5 m/s`
- 概念题正确解释金属和木头的导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

模型每次仍只接收当前问题，没有多轮记忆。当前未实现图片输入、数据库、RAG、工具调用、
MCP、微调或真正的多轮上下文，也未加入其他 Agent 框架。
