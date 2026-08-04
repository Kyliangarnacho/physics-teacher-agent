# 初中物理教师 Agent

## 项目定位

本项目是“初中物理教师 Agent + 阿里云百炼千问 API”。当前已完成 Stage 09：在既有
Router、RAG、Tool、有限 Retry 和 Trace 基础上，加入图片视觉理解、可选 OCR、多图片聊天和
人工确认流程。当前文本模型为 `qwen3.7-flash`，提示词版本仍为
`teacher_v3_personal_humor`。

## 环境与配置

- Python 3.12 虚拟环境：`.venv`
- 主要依赖：`openai`、`python-dotenv`、`streamlit`、`pydantic`、`Pillow`、`jieba`、
  `rank-bm25`
- 本地配置：`.env`
- 当前模型：`qwen3.7-flash`

`.env` 保存本地 API Key、Base URL 及文本/视觉/OCR 模型配置，已被 Git 忽略且不会提交。
可复制 `.env.example` 后填写本地配置；OCR 模型未配置时会安全跳过 OCR 增强。

## 代码职责

- `app.py`：提供 Streamlit 文本/多图片聊天页面，调用视觉服务和统一 Agent，并保存安全的
  回答、决策、来源、工具记录和 Trace
- `main.py`：运行固定平均速度题，作为终端回归入口
- `src/config.py`：加载并校验 `.env` 中的千问配置
- `src/schemas.py`：定义 `TeachingMode`、带 `calculation_required` 的
  `QuestionAnalysis` 和带 `use_tools` 的 `RouteDecision`
- `src/analyzer.py`：调用千问分析问题并校验结构化 JSON，失败时返回安全 fallback
- `src/router.py`：处理教学模式覆盖、RAG 三态、工具路由和缺图安全拦截
- `src/agent.py`：统一编排 Analyzer、Router、可选 RAG、可选 Tool Client 和最终回答
- `src/prompts.py`：保存教师 Prompt、Analyzer Prompt 和四种模式指令
- `src/model_client.py`：按顺序注入模式指令和可选参考资料，调用千问并返回回答
- `knowledge/physics_notes_v1.jsonl`：保存 10 条可检索的初中物理知识卡片
- `src/retriever.py`：使用 `jieba` 分词和 `BM25Okapi` 排序，过滤低相关结果
- `src/rag.py`：提供纯检索 context/sources 接口，并保持旧 RAG 回答入口兼容
- `src/tools/physics_calculators.py`：实现五个基于 `Decimal` 的本地物理计算函数
- `src/tools/schemas.py`：定义五类严格 Pydantic v2 工具参数 Schema
- `src/tools/registry.py`：显式注册五个白名单工具，校验参数并返回结构化执行记录
- `src/tool_client.py`：执行单工具、两轮 Qwen Function Calling，并返回工具记录
- `src/observability.py`：构建步骤 Trace、整次 AgentRunTrace，并统一记录安全错误类型
- `src/retry.py`：提供至多重试一次的通用有限 Retry，不负责业务错误分类
- `src/vision/`：负责图片内存预处理、视觉提取、可选 OCR、结果合并、多图 Batch 与安全上下文
- `src/ui/paste_images.py`：提供图片粘贴辅助与哈希去重；页面同时保留原生附件上传能力
- `scripts/probe_stage09_vision.py`：人工验证视觉模型、多模态消息和结构化提取能力
- `scripts/probe_stage07_function_calling.py`：人工验证底层 Function Calling 兼容链路
- `scripts/probe_stage07_agent_e2e.py`：人工验证统一 Agent 的真实工具端到端链路
- `scripts/validate_knowledge_base.py`：校验知识卡片的 JSON、字段、类型和重复 ID
- `evaluation/stage04_text_cases_v1.json`：保存 15 道 Stage 04 纯文本评测题
- `evaluation/results/`：保存不同提示词版本的 JSONL 评测结果
- `evaluation/stage08_agent_cases_v1.json`：保存 8 道 Stage 08 代表性 Agent 路径题
- `evaluation/run_stage08_agent_evaluation.py`：支持 fake/real、筛选与断点续跑的 Agent 评测入口
- `evaluation/summarize_stage08_results.py`：汇总路由、请求、Retry、RAG、工具和错误统计
- `evaluation/reviews/`：保存人工评审表与版本对比
- `evaluation/validate_stage04_text_cases.py`：校验题目字段、数量和 ID
- `evaluation/run_stage04_text_evaluation.py`：支持 `--limit` 和断点续跑的评测入口
- `tests/`：使用 Python 标准库 `unittest` 验证 Schema、Analyzer、Router、Agent、
  检索、RAG、工具计算、Registry、Tool Client、Trace、Retry、视觉/OCR、多图页面和评测 Runner

## Stage 09 图片理解与多图聊天

Stage 09 当前实现：

- 支持 JPEG、PNG、WEBP 图片的内存校验、EXIF 方向处理、尺寸约束和安全 Data URL 请求构造
- Vision Client 只提取题意与视觉关系；OCR 为可选增强，不负责解题或猜测图形连接
- 聊天输入支持纯文字、附件和 Ctrl+V 图片粘贴，一次最多 3 张图片，并按 SHA-256 去重
- 每张图独立识别并按上传顺序组成 Batch；清晰且无不确定项时自动进入 Agent
- `needs_confirmation` 或存在不确定项时暂停回答，让用户编辑并一次确认；`unreadable` 阻止提交
- 确认后的安全图片上下文复用现有 Analyzer、Router、RAG、Tool、Retry 与 Trace 链路
- 会话消息只保存文件名、哈希、Batch/视觉运行 ID 等安全元数据，不保存原图、Data URL、
  Base64 或完整图片上下文

## Stage 08 Trace、有限 Retry 与评测

Stage 08 当前实现：

- `StepTrace` 与 `AgentRunTrace` 记录步骤状态、尝试次数、耗时、模型请求数和安全错误摘要
- Analyzer、工具选择和工具结果回传的 API 调用最多重试一次；JSON 解析、Schema、协议和本地
  工具错误不重试
- 工具结果回传重试复用已经生成的工具记录和消息，不重复执行本地工具
- 统一 Agent 汇总 Analyzer、Router、RAG、普通回答或三个工具步骤的运行轨迹
- Streamlit 在每条新 Assistant 消息中保存 Trace，并通过“Agent 运行轨迹”折叠区展示摘要与步骤
- Stage 08 Runner 包含恰好 8 道代表题，支持 `fake`/`real`、`--limit`、`--case-id`、JSONL
  断点续跑和 Markdown 汇总

真实评测已按固定 Runner 流程完整运行一次，8 个唯一 case 均有非空回答、路由和 Trace：7 题
`completed`、1 题缺图 `blocked`、0 题失败、0 次 fallback、0 个 Retry 成功步骤。预期路由匹配
5/8；3 个偏差分别是 S08-AGENT-001 与 S08-AGENT-007 自动启用了 RAG，以及
S08-AGENT-002 被判断为 `explain` 而非预期的 `solve`。这些结果原样保留，没有为提高分数改写
Prompt、路由规则或工具算法。

## Stage 07 本地工具与统一 Agent

Stage 07 当前实现：

- 五个本地工具：`calculate_average_speed`、`calculate_density`、
  `calculate_ohms_law`、`calculate_electric_power`、`convert_physics_unit`
- 五类 Pydantic v2 参数 Schema，拒绝额外字段、非法数值和不合法参数组合
- 显式白名单 Tool Registry，不根据模型字符串动态执行任意函数
- 仅对已登记数值字段的纯 JSON 数字字符串进行受控规范化
- Analyzer 输出 `calculation_required`，Router 输出 `use_tools`
- `solve`、`explain`、`hint`、`diagnose` 四种教学模式
- `auto`、`force`、`off` 三态 RAG 策略
- 缺图时停止检索和回答，提示用户补充题图或完整描述
- Qwen Function Calling 工具选择、本地执行和 `role=tool` 结果回传
- 普通、RAG、Tool、RAG+Tool 四条统一 Agent 路径
- Streamlit 页面展示 Agent 决策、知识来源和本地计算工具记录

当前完整数据流：

```text
用户问题
→ Analyzer 第一次模型调用
→ QuestionAnalysis 校验
→ Router
→ RouteDecision
→ 可选 BM25 RAG，统一准备 context 和 sources
→ use_tools=false：普通最终回答模型
→ use_tools=true：工具选择模型 → 白名单 Registry 本地执行 → role=tool 结果回传模型
→ 页面展示答案、Agent 决策、来源和可选工具记录
```

Analyzer 返回非法 JSON、字段不合法或调用异常时不重试，而是使用安全的 `solve`
分析继续处理，并在最终结果中标记 `analysis_fallback=true`。工具题正常情况下包含三次模型
请求：Analyzer 一次、Tool Client 工具选择一次、工具结果回传一次；Registry 的计算在本地
完成，不属于模型请求。

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

页面侧边栏提供两组选择，默认均为自动：

- 教学模式：自动判断、完整解题、概念讲解、只给提示、错误诊断。
- 知识库策略：自动决定、强制使用、不使用。
- 自动决定或强制使用 RAG 时，最多选取 3 条达到相对阈值的资料。
- 有来源：在回答下方的折叠区域显示知识卡片 ID、主题和原始文件名。
- 无来源：提示本次按普通问答处理。
- 每条新回答还保存并显示最终教学模式、物理主题、问题类型、RAG 决策、简短理由和
  Analyzer fallback 状态。
- Analyzer 判断需要确定性计算且条件完整时，Router 可启用本地工具；回答下方的
  “本地计算工具记录”展示工具名、参数、规范化字段、状态、公式、结果、单位和安全错误
  摘要。

本地校验与单元测试：

```powershell
.venv\Scripts\python.exe scripts\validate_knowledge_base.py
.venv\Scripts\python.exe evaluation\validate_stage04_text_cases.py
.venv\Scripts\python.exe evaluation\validate_stage08_agent_cases.py
.venv\Scripts\python.exe -m unittest discover -v
.venv\Scripts\python.exe -m compileall app.py main.py src scripts tests evaluation
.venv\Scripts\python.exe -m pip check
```

Stage 08 Runner 默认使用完全不调用 API 的 fake 模式；真实模式必须显式指定：

```powershell
.venv\Scripts\python.exe evaluation\run_stage08_agent_evaluation.py --mode fake --limit 2
.venv\Scripts\python.exe evaluation\run_stage08_agent_evaluation.py --mode real `
  --output evaluation\results\stage08_agent_real_v1.jsonl
.venv\Scripts\python.exe evaluation\summarize_stage08_results.py `
  evaluation\results\stage08_agent_real_v1.jsonl `
  --markdown evaluation\results\stage08_agent_real_v1_summary.md
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

网页使用 `st.session_state` 保存当前页面的用户消息和教师回答。新 Assistant 消息保存
`content`、`sources`、`analysis`、`route`、`analysis_fallback`、`tool_records`、
`tool_model_requests` 和 `trace`，因此同一会话内脚本重新运行时仍能显示当次决策、来源、
工具记录和运行轨迹。
旧消息缺少这些字段时也能安全显示。

历史仅属于当前浏览器会话/标签页。浏览器新会话、硬刷新导致会话重建或服务重启后，
历史不会持久保存。

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
- 当前全部单元测试为 393 项，全部通过
- Stage 05 已完成普通网页问答和真实千问调用，页面与终端没有出现应用 traceback
- 已真实验证电热器和凸透镜 RAG 问答；加入相对分数过滤后再次验证电热器问题，网页来源
  只返回 `KB-POWER-001`
- 概念验收已确认核心 RAG 数据流；Mock 测试边界和 BM25 分数含义仍需继续复习，但不阻塞
  当前工程验收
- Stage 06 已完成真实 `diagnose` 和 `explain` Agent 场景；自动 Agent 问答通常包含
  Analyzer 和最终回答两次模型调用
- 教师 Prompt 已加入绝对化前提检查、先勘误再回答和条件变化事实守护；`KB-ELEC-003`
  已补充额定功率定义、相等条件和白炽灯丝电阻随温度变化的边界
- Stage 07 底层 Function Calling 探针已验证 Qwen 接受五个工具 Schema、返回工具调用并
  接受匹配的 `role=tool` 消息
- Stage 07 统一 Agent 真实 E2E 使用“12 V、6 Ω 求电流”问题，Analyzer 判断需要计算，
  Router 启用 `calculate_ohms_law`，Registry 本地得到 `2 A`，最终教师回答非空；本次链路
  模型请求数为 3，且没有误调用普通回答函数
- Stage 08 的 8 道真实 Agent 评测只运行一轮：7 题完成、1 题缺图拦截、0 题失败，预期路由
  匹配 5/8；总模型请求 18 次、RAG 检索 4 次、本地工具执行 3 次，无 fallback、Retry 或空回答
- Stage 08 Streamlit Trace 专项测试通过，服务启动检查返回 HTTP 200
- Stage 09 单图、多图、可选 OCR、自适应确认、图片上下文进入 RAG/Tool、附件与粘贴去重均有
  本地测试覆盖；完整回归为 393/393

用户手动验证的网页结果：

- 计算题返回 `5 m/s`
- 概念题正确解释金属和木头的导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

知识库只有 10 条文本卡片，覆盖范围有限，不能代表完整初中物理。当前 BM25 主要依赖词面
匹配，没有向量检索、重排序、系统化召回评测或自动事实校验。当前真实网页验收只覆盖普通
问答、电热器和凸透镜等少量问题，不能代表完整 RAG 效果。

非工具题的自动模式通常需要 Analyzer 和最终回答两次模型请求；工具题通常需要 Analyzer、
工具选择和工具结果回传三次模型请求，会进一步增加延迟和 API 费用。本地 Registry 执行
不属于模型请求。有限 Retry 会在可重试 API 异常时额外增加一次请求。Analyzer 仍可能发生
语义分类、RAG 或计算需求判断错误；Schema 和 Router 只能校验结构与流程，不能验证复杂题目
的物理建模是否正确，模型仍可能遗漏预期细节。

Stage 08 的 8 道题只覆盖代表性工程路径，不是完整初中物理能力评测，也没有对全部答案做
系统人工物理评分。真实运行的 5/8 路由匹配说明自动分类仍有偏差，当前结果不能证明所有题型
都能稳定选择预期模式或知识库策略。

模型每次仍只接收当前问题，没有真正的多轮上下文记忆；页面显示历史不等于模型记忆。
当前工具链一次只允许执行一个工具，不支持多个或并行工具调用，也不支持工具调用循环。
图片理解依赖视觉模型输出，模糊文字、手写内容和复杂图形仍可能需要人工确认；一次提交最多
3 张图片，OCR 仅在已配置可用模型时启用。知识库仍只有 10 条文本卡片，BM25 覆盖有限。
当前未实现数据库、长期记忆、MCP、微调或通用的跨图片语义推理。
