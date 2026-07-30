# Physics Teacher Agent

## 项目定位

本项目是“初中物理教师 Agent + 阿里云百炼千问 API”。当前已实现 Stage 05 最小
RAG：可以选择使用本地初中物理知识卡片，通过 BM25 检索相关资料后再调用
`qwen3.7-flash`。提示词版本仍冻结为 `teacher_v3_personal_humor`。

## 环境与配置

- Python 3.12 虚拟环境：`.venv`
- 主要依赖：`openai`、`python-dotenv`、`streamlit`、`jieba`、`rank-bm25`
- 本地配置：`.env`
- 当前模型：`qwen3.7-flash`

`.env` 保存 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL` 和 `QWEN_MODEL`，已被 Git
忽略且不会提交。可复制 `.env.example` 后填写本地 API Key。

## 代码职责

- `app.py`：提供 Streamlit 聊天页面，在普通问答和本地知识库 RAG 之间切换，并保存
  当前页面的消息与来源
- `main.py`：运行固定平均速度题，作为终端回归入口
- `src/config.py`：加载并校验 `.env` 中的千问配置
- `src/prompts.py`：保存提示词版本和初中物理教师 system prompt
- `src/model_client.py`：构造普通消息或带参考资料的消息，调用千问并返回回答
- `knowledge/physics_notes_v1.jsonl`：保存 10 条可检索的初中物理知识卡片
- `src/retriever.py`：使用 `jieba` 分词和 `BM25Okapi` 排序，过滤低相关结果
- `src/rag.py`：编排知识检索、context 构造、模型调用和精简来源返回
- `scripts/validate_knowledge_base.py`：校验知识卡片的 JSON、字段、类型和重复 ID
- `evaluation/stage04_text_cases_v1.json`：保存 15 道 Stage 04 纯文本评测题
- `evaluation/results/`：保存不同提示词版本的 JSONL 评测结果
- `evaluation/reviews/`：保存人工评审表与版本对比
- `evaluation/validate_stage04_text_cases.py`：校验题目字段、数量和 ID
- `evaluation/run_stage04_text_evaluation.py`：支持 `--limit` 和断点续跑的评测入口
- `tests/`：使用 Python 标准库 `unittest` 验证评测、检索、RAG 和消息构造

## Stage 05 最小 RAG

知识库当前包含 10 条卡片，覆盖欧姆定律与动态电路、电功率、光学、实验与易错点。检索器
将每张卡片的 `chapter`、`topic`、`keywords` 和 `content` 合并后分词建索引。

默认检索参数为 `top_k=3` 和 `min_score_ratio=0.3`：先保留 BM25 正分结果，再删除低于
最高分 30% 的卡片，最后限制返回数量。完全没有正相关分数时返回空结果。

普通问答数据流：

```text
用户问题 → answer_question → 基础 system prompt + 原问题 → 千问回答
```

RAG 问答数据流：

```text
用户问题 → KnowledgeRetriever → 相关知识卡片 → build_context
        → answer_question(question, context) → 千问回答 + 精简 sources
```

参考资料以独立 system 消息插入基础教师提示词和用户问题之间，只作为物理知识依据；消息
明确禁止执行资料中的指令，并要求资料不足时不得编造。没有检索结果时使用
`context=None`，自动退回普通问答，来源列表为空。

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

页面中的“使用本地物理知识库”复选框默认不勾选：

- 不勾选：保持原有普通问答。
- 勾选：每次使用本地 BM25 检索，最多选取 3 条达到相对阈值的资料。
- 有来源：在回答下方的折叠区域显示知识卡片 ID、主题和原始文件名。
- 无来源：提示本次按普通问答处理。

本地校验与单元测试：

```powershell
.venv\Scripts\python.exe scripts\validate_knowledge_base.py
.venv\Scripts\python.exe evaluation\validate_stage04_text_cases.py
.venv\Scripts\python.exe -m unittest discover -v
.venv\Scripts\python.exe -m compileall app.py main.py src scripts tests
.venv\Scripts\python.exe -m pip check
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

网页使用 `st.session_state` 保存当前页面的用户消息和教师回答。RAG 消息还保存
`rag_enabled` 和精简的 `sources`，因此页面重新运行时仍能显示当次来源。历史仅属于
当前浏览器会话/标签页，关闭页面或重启服务后不会持久保存。

页面可以显示多轮聊天记录，但每次调用模型时只发送当前问题，不发送页面中的完整历史，
因此目前不是真正的多轮上下文对话。RAG context 也只来自当前问题的本次检索。

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
- Stage 05 知识库 10 条卡片通过结构与唯一性校验
- 检索、RAG 编排和 context 消息均由 fake/mock 测试覆盖，不调用千问
- 当前全部单元测试为 29 项，全部通过
- Stage 05 已完成普通网页问答和真实千问调用，页面与终端没有出现应用 traceback
- 已真实验证电热器和凸透镜 RAG 问答；加入相对分数过滤后再次验证电热器问题，网页来源
  只返回 `KB-POWER-001`
- 概念验收已确认核心 RAG 数据流；Mock 测试边界和 BM25 分数含义仍需继续复习，但不阻塞
  当前工程验收

用户手动验证的网页结果：

- 计算题返回 `5 m/s`
- 概念题正确解释金属和木头的导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

知识库只有 10 条文本卡片，覆盖范围有限，不能代表完整初中物理。当前 BM25 主要依赖词面
匹配，没有向量检索、重排序、系统化召回评测或自动事实校验。当前真实网页验收只覆盖普通
问答、电热器和凸透镜等少量问题，不能代表完整 RAG 效果。

模型每次仍只接收当前问题，没有多轮记忆。当前未实现图片输入、OCR、数据库、工具调用、
MCP、微调或真正的多轮上下文，也未加入其他 Agent 框架。
