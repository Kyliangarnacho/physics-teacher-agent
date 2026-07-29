# 当前状态

## Stage 04 状态

“初中物理教师 Agent + 阿里云百炼千问 API”当前使用 `qwen3.7-flash`，提示词版本为
`teacher_v3_personal_humor`。本阶段决定冻结该版本，不再根据现有 5 道题继续调整。

当前个人风格以讲解逻辑、条件分类、因果链和纠错方式为核心。幽默仅在语境自然匹配时
偶尔出现，不要求每题都有。

## 当前结构

- `app.py`：Streamlit 页面入口，负责聊天输入、加载提示、消息显示和当前页面历史
- `main.py`：固定平均速度题的终端回归入口
- `src/config.py`：使用 `python-dotenv` 加载并校验千问配置
- `src/prompts.py`：定义 `PROMPT_VERSION = "teacher_v3_personal_humor"` 和教师
  system prompt
- `src/model_client.py`：从提示词模块导入 system prompt，并使用 OpenAI 兼容客户端调用千问
- `evaluation/stage04_text_cases_v1.json`：15 道纯文本评测题
- `evaluation/results/`：v2 与 v3 的前 5 题 JSONL 结果
- `evaluation/reviews/`：人工评审表和新旧版本对比
- `evaluation/validate_stage04_text_cases.py`：评测数据校验入口
- `evaluation/run_stage04_text_evaluation.py`：支持 `--limit`、独立输出文件和断点续跑
- `tests/`：基于标准库 `unittest` 的评测工具测试

API Key 仅保存在本地 `.env` 中；`.env` 和 `.venv` 均被 Git 忽略。
原始资料和私有风格示例分别保存在 `data/raw/` 与 `data/private_evaluation/`，两个目录
均被 Git 忽略且不会提交。

## 运行命令

```powershell
# 终端回归
.venv\Scripts\python.exe main.py

# 网页启动
.venv\Scripts\python.exe -m streamlit run app.py

# 校验 Stage 04 评测题
.venv\Scripts\python.exe evaluation\validate_stage04_text_cases.py

# 运行全部本地单元测试（不调用 API）
.venv\Scripts\python.exe -m unittest discover -v
```

## 会话行为

`st.session_state` 只保存当前浏览器会话/标签页中的消息，页面关闭或服务重启后不会持久化。
当前页面可以显示历史消息，但每次模型请求只发送当前问题，不发送完整页面历史。

## 验证状态

工程与评测状态：

- `qwen3.7-flash` 终端回答包含 `5 m/s`
- 空问题被直接拦截，未加载配置或调用模型
- 临时缺少 `QWEN_MODEL` 时错误清楚，测试进程结束后正常配置仍为 `qwen3.7-flash`
- Streamlit 启动检查返回 HTTP 200
- `compileall`、`pip check`、`git diff --check` 通过
- 15 道 Stage 04 文本题字段完整且 ID 唯一
- `teacher_v2_personal` 与 `teacher_v3_personal_humor` 均完成前 5 道真实调用
- 两组前 5 题结果均为 `status=ok` 且回答非空
- 用户对 v2 的 S04-TXT-001 人工评分为 7/8
- 当前只完成前 5 道基线，未运行全部 15 道

用户已手动验证：

- 网页计算题返回 `5 m/s`
- 网页概念题正确解释金属和木头导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

当前小样本不足以证明 v3 全面优于 v2，也不代表完整的初中物理能力。自动预检查只验证
工程数据，不替代物理正确性和教学效果的人工评分。

模型每次只接收当前问题，没有多轮记忆。尚未实现图片、数据库、RAG、工具调用、MCP、
微调和真正的多轮上下文。
