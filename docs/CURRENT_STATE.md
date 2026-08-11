# 当前状态

## Stage 11.3 状态

“初中物理教师 Agent + 阿里云百炼千问 API”当前使用 `qwen3.7-flash`，提示词版本仍为
`teacher_v3_personal_humor`。Stage 10 在 Stage 09 视觉/多图基础上加入 SQLite 持久会话、
多会话管理、最近历史与教学状态，以及长期学习记忆的手动提取、确认和召回；Stage 11.1
进一步加入有限多工具编排、参数 repair、Model-only fallback 和公开工具选择评测。Stage 11.2
加入 SQLite Generation Job、双 Worker 后台生成、恢复/Retry、图片附件与多图上下文。Stage 11.3
现已加入 Migration V4、Rolling Summary、Unsummarized Bridge、conversation-scoped BM25 历史
检索、确定性 Context Budget、单轮 ContextBundle 快照、Analyzer/Tool/Final Projection、后台
Summary maintenance 与只含安全统计的 Context Trace。

当前个人风格以讲解逻辑、条件分类、因果链和纠错方式为核心。幽默仅在语境自然匹配时
偶尔出现，不要求每题都有。

## 当前结构

- `app.py`：Streamlit 文本/多图片聊天入口；视觉识别与人工确认在页面线程完成，确认后的教师
  回答进入后台 Generation Job
- `main.py`：固定平均速度题的终端回归入口
- `src/config.py`：使用 `python-dotenv` 加载并校验千问配置
- `src/schemas.py`：定义教学模式、带计算需求的分析结果和带工具决策的路由结果
- `src/analyzer.py`：执行问题分析；API 调用异常可有限重试，解析或 Schema 失败时安全 fallback
- `src/router.py`：处理模式覆盖、RAG 策略、工具启用、缺图拦截和回答许可
- `src/agent.py`：统一编排分析、路由、可选 RAG、可选 Tool Client、最终回答和运行 Trace
- `src/prompts.py`：定义教师 Prompt、Analyzer Prompt 和 `MODE_INSTRUCTIONS`
- `src/model_client.py`：注入模式指令和可选参考资料，并调用最终回答模型
- `knowledge/physics_notes_v1.jsonl`：10 条初中物理知识卡片
- `src/retriever.py`：使用 `jieba` 和 `rank_bm25.BM25Okapi` 建立本地文本索引
- `src/rag.py`：提供纯检索 context/sources 接口，并保留原 RAG 回答入口
- `src/tools/physics_calculators.py`：七个基于 `Decimal` 的确定性本地计算函数
- `src/tools/schemas.py`：七类严格 Pydantic v2 工具参数合同
- `src/tools/registry.py`：七工具白名单、数值字符串规范化、参数校验和结构化执行记录
- `src/tool_client.py`：以 `tool_choice=auto` 进行有限多工具 Function Calling，支持 1～5 个
  调用的顺序执行、partial success、一次参数 repair 与 Model-only fallback；API 调用可有限
  重试且不会重复执行已成功的本地工具
- `src/observability.py`：构建步骤 Trace、整次 AgentRunTrace，并定义统一安全错误类别
- `src/retry.py`：提供至多重试一次的通用有限 Retry
- `src/vision/`：图片预处理、视觉/OCR Client、结果合并、安全上下文、多图 Batch 与关系融合
- `src/ui/paste_images.py`：图片粘贴辅助和哈希去重；原生附件上传仍可独立使用
- `src/storage/`：SQLite 短连接、Migration V1～V4、Generation Job、Conversation Summary 与 Repository
- `src/conversation/`：会话 Schema、最近历史窗口、教学状态解析，以及 enqueue/execute 分离的
  Conversation Service；旧同步入口继续保留
- `src/context/`：Summary/Bridge/Recent、旧历史 BM25、Context Budget、ContextBundle 与三类 Projection
- `src/tasks/`：固定两个 Worker 的 GenerationTaskManager，以及独立单 Worker 的学习表现分析和
  Context Maintenance 管理器
- `src/memory/`：长期记忆候选提取、确认、检索、模型注入与异步学习表现分析
- `scripts/probe_stage09_vision.py`：视觉模型结构化提取能力的人工诊断探针
- `scripts/probe_stage10_conversation_context.py` / `probe_stage10_conversation_service.py`：
  历史/状态注入与两轮会话的真实探针
- `scripts/probe_stage10_memory_candidates.py` / `probe_stage10_memory_retrieval.py` /
  `probe_stage10_memory_page.py`：长期记忆候选、检索与页面交互的真实探针
- `scripts/probe_stage11_2_background_jobs.py`：不调用 API 的双会话 Fake Slow 后台任务验收入口
- `scripts/probe_stage07_function_calling.py`：底层 Function Calling 人工诊断探针
- `scripts/probe_stage07_agent_e2e.py`：统一 Agent 真实端到端验收探针
- `scripts/validate_knowledge_base.py`：校验知识库 JSONL、字段、类型和 ID 唯一性
- `evaluation/stage04_text_cases_v1.json`：15 道纯文本评测题
- `evaluation/results/`：保存 Stage 04 结果及 Stage 08 真实 JSONL 与 Markdown 汇总
- `evaluation/stage08_agent_cases_v1.json`：8 道代表性 Agent 路径题
- `evaluation/run_stage08_agent_evaluation.py`：支持 fake/real、筛选和断点续跑的评测入口
- `evaluation/summarize_stage08_results.py`：汇总路由、请求、Retry、RAG、工具和错误统计
- `evaluation/stage11_tool_selection_cases_v1.json`：24 条公开工具适用性评测题
- `evaluation/run_stage11_tool_selection_evaluation.py`：真实但只评估工具选择的专项 Runner
- `evaluation/results/stage11_tool_selection_comparison.md`：公开策略对比结果
- `evaluation/reviews/`：人工评审表和新旧版本对比
- `evaluation/validate_stage04_text_cases.py`：评测数据校验入口
- `evaluation/run_stage04_text_evaluation.py`：支持 `--limit`、独立输出文件和断点续跑
- `tests/`：基于标准库 `unittest` 的分析、路由、RAG、工具、Agent、Trace、Retry、
  视觉/OCR、多图交互、页面展示和评测 Runner 测试

API Key 仅保存在本地 `.env` 中；`.env` 和 `.venv` 均被 Git 忽略。
原始资料和私有风格示例分别保存在 `data/raw/` 与 `data/private_evaluation/`，两个目录
均被 Git 忽略且不会提交。

## Stage 11.3 上下文数据流

```text
Messages
→ Rolling Summary / Unsummarized Bridge / Recent 3 turns
→ conversation-scoped History Retrieval
→ Context Manager + deterministic character Budget
→ one ContextBundle snapshot per Generation
→ Analyzer / Tool / Final Projection
→ Agent
```

Summary 是有损压缩，Bridge 保存已经离开 Recent 但尚未被 Summary boundary 覆盖的完整原文轮次，
因此 `Summary + Bridge + Recent` 始终连续。Retrieved 只从 Summary-covered 的更老轮次中召回，排除
Bridge 和 Recent；旧 assistant 仅代表过去谈过什么，不是可靠物理知识。当前用户问题始终是本轮
唯一任务，历史只用于消歧、条件恢复、指代理解和连续性。

```text
Generation completed
→ O(1) Context maintenance submit
→ single ContextMaintenance Worker
→ persisted stale check
→ Rolling Summary refresh
```

Summary maintenance 不进入 Generation critical path，不建立 summary_jobs。失败时旧 Summary、covered
boundary 和 Bridge 均保持不变，下次 stale 时可以再次调度。

阶段实现与调试边界见 `docs/STAGE11_3_CONTEXT_ENGINEERING.md`。

## 图片数据流

```text
文字与最多 3 张图片（附件或 Ctrl+V）
→ 图片内存预处理与同批去重
→ 每张图片独立 Vision 提取
→ 按需执行可选 OCR 并合并
→ 多图关系判断：同题融合或独立题分别保留
→ 清晰结果自动采用；不确定结果等待一次人工确认
→ 按图片顺序构造已确认上下文
→ 原子保存 user 与 pending Generation Job
→ 后台 Worker 恢复已确认上下文并执行 Analyzer → Router → 可选 RAG / Tool → 最终回答
→ 页面展示安全识别元数据、来源、工具记录和 Trace
```

原图以受控本地附件文件保存，SQLite 消息元数据仅保存附件 ID、相对路径、MIME、文件名和哈希；
图片原始字节、Data URL、Base64 和完整确认上下文不会写入 Generation Job、聊天元数据或 Trace。
上传新 Batch 或完成一次发送后，不会让上一批图片自动进入下一轮纯文字问题；但 Analyzer 判断为
follow-up/uncertain 时，可安全继承上一题已确认的图片文本上下文。

## 运行命令

```powershell
# 终端回归
.venv\Scripts\python.exe main.py

# 网页启动
.venv\Scripts\python.exe -m streamlit run app.py

# 校验 Stage 04 评测题
.venv\Scripts\python.exe evaluation\validate_stage04_text_cases.py

# 校验 Stage 05 知识库
.venv\Scripts\python.exe scripts\validate_knowledge_base.py

# 校验 Stage 08 代表题（不调用 API）
.venv\Scripts\python.exe evaluation\validate_stage08_agent_cases.py

# 运行全部本地单元测试（不调用 API）
.venv\Scripts\python.exe -m unittest discover -v
```

Stage 08 Runner 默认为 fake 模式；`--mode real` 才会调用当前 Agent 和模型。结果以 JSONL
断点续跑，并可通过 `evaluation/summarize_stage08_results.py` 输出终端与 Markdown 汇总。

## Agent 数据流

```text
用户问题
→ Analyzer 第一次模型调用
→ QuestionAnalysis 校验
→ Router
→ RouteDecision
→ 可选 RAG，统一准备 context 和 sources
→ 普通路径：最终回答模型
→ RAG 路径：context + 最终回答模型
→ Tool 路径：工具选择模型（可选 1～5 个调用）→ Registry 顺序执行 → 全部 role=tool 结果回传
→ RAG+Tool 路径：同一次检索 context → 有限多工具选择、本地执行和结果回传
→ 汇总步骤 Trace 与 AgentRunTrace
→ 页面展示答案、决策、来源、可选工具记录和运行轨迹
```

四种教学模式为 `solve`、`explain`、`hint`、`diagnose`。页面可自动采用 Analyzer
模式，也可手动覆盖。RAG 策略为 `auto`、`force`、`off`：

- `auto` 使用 Analyzer 的 `needs_rag`
- `force` 强制执行本地 BM25 检索
- `off` 完全跳过检索

检索仍使用 `jieba + BM25Okapi`，默认 `top_k=3`、`min_score_ratio=0.3`。模式指令
位于教师 system prompt 之后，RAG 参考资料位于模式指令之后，用户问题始终放在最后。

没有正相关资料时，context 使用 `None`，仍调用普通 `answer_question`，sources 为空。
`image_required=true` 时 Router 设置 `should_answer=false`，关闭 RAG 和最终模型调用，
只提示用户补充题图或完整描述。Analyzer 失败时使用安全默认分析，并在结果中记录
`analysis_fallback=true`。

Analyzer、工具选择和工具结果回答的 API 调用异常最多重试一次；JSON 解析、Schema、协议、
参数校验和本地工具错误不重试。工具结果回答重试复用同一份工具记录，不再次执行本地工具。
每次 Agent 运行汇总总模型请求数、RAG 检索数、本地工具执行数以及各步骤状态和安全错误摘要；
Trace 还记录 repair、partial success 与 Model-only fallback。

Analyzer 的 `calculation_required=true` 且条件完整时，Router 设置 `use_tools=true`。
七个白名单工具分别计算平均速度、密度、欧姆定律、电功率、单位换算、机械功率和滑轮组效率。
工具选择使用 `tool_choice=auto`：只有白名单工具能直接、可靠核算关键数值时才主动调用；
即使可心算也保留有价值的确定性核算；无匹配工具时正常 Model-only，no-tool 不视为错误。
多个独立且被覆盖的计算可一次返回多个调用。工具参数先经对应 Pydantic Schema 校验；Qwen
返回的纯 JSON 数字字符串只在已登记数值字段中受控转换，其余字符串、额外字段和非法参数
不会被猜测执行。

普通、RAG、Tool、RAG+Tool 四条路径共用 `run_teacher_agent()`。单个 Tool Batch 最多执行
5 个调用，按模型顺序运行；部分调用失败时保留全部记录，存在任一成功记录仍可生成最终答案。
参数合同错误最多进行 1 次批量 repair；全失败、无调用或协议超限时回退 Model-only。
非工具题通常包含 Analyzer 和最终回答两次模型请求；工具题通常包含 Analyzer、工具选择、
工具结果回传三次模型请求，repair 或有限 Retry 会增加请求。Registry 的参数校验和物理计算
完全在本地执行，不属于模型请求。

## 会话与持久化

SQLite 是持久消息的 Source of Truth，`st.session_state` 只保存当前会话 ID、会话列表
缓存和未发送图片、待确认图片、输入控件等页面临时状态。支持多会话新建、切换、重命名、
清空（保留会话）与删除；浏览器新会话或服务重启后，历史从 SQLite 恢复。

页面使用 `list_messages` 恢复历史并用 `display_content` 渲染；assistant 的 sources、route、
tool_records、trace 从对应 agent_run 的 JSON 字段还原。每个 Generation 只构造一次 ContextBundle，
以 12000 字符的项目内部安全预算组合 Summary、Bridge、Recent、Retrieved、ConversationState 与
Learning Memory，再为 Analyzer、Tool 和 Final 生成确定性 Projection；当前问题始终是最后一条
user 消息和本轮唯一执行目标。

后台回答状态为 `pending → running → completed/failed`；服务启动时遗留 `running` 会标记为
`interrupted`，`pending` 会自动重新调度。`failed/interrupted` 只能人工 Retry，且复用原 Job 与
原 user 消息。Streamlit 使用进程级缓存的 Task Manager 和只读 fragment Polling：仅当前会话
存在活动 Job 时轮询，终态后停止并刷新消息。不同会话按 `conversation_id` 隔离，可由两个
Worker 并发生成；活动会话可重命名，但不能删除或清空。

Generation Job 只保存回答所需文本参数和已确认图片上下文。严格 Schema 拒绝 bytes、额外字段、
未确认图片上下文、Data URL 与明显 Base64；原图、UploadedFile、Data URL 和 Base64 不进入 Job。
Worker 不访问 `st.*` 或 `session_state`，每次 Repository 操作独立打开 SQLite 短连接；进程内
`scheduled` 集合与数据库原子 claim 共同防止重复执行。

Generation Future 的未处理异常会将仍为 pending/running 的 Job 安全落成 failed；进程重启时遗留
running 仍转为 interrupted。两种终态都可复用原 Job 与原 user message 人工 Retry。completed 后只
向 Context Maintenance Worker 轻量 submit，stale scan 和 Summary API 均不占用 Generation Worker。

## 长期记忆流程

- 成功回答下方可点击“分析本轮学习表现”手动提取候选（不自动提取）；提取在独立后台线程执行，
  不占用 Generation Job 的两个 Worker，也不会阻塞继续提问或切换会话；
- 候选暂存 `session_state`（按会话与消息隔离），确认保存后才写入 learning_memories；
- 只有 `confirmed=1` 且 `active=1` 的记忆会被召回并注入相关题目；
- 普通单次错误不自动认定为 weakness；misconception 需要明确错误规律，preference 需要用户明确表达；
- Context Manager 只检索并格式化现有 Memory，失败时安全降级为空；Tool Projection 不携带 Memory；
- Conversation Summary 记录本会话事实连续性，Learning Memory 记录用户确认的跨会话稳定学习信息，
  两者不会互相写入或替代；
- “学习档案”展示已确认记忆，支持停用和删除；
- 提取/确认失败显示安全错误，不泄露密钥、SQL 或路径。

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
- Stage 05 知识库包含 10 条卡片，结构校验通过
- BM25 默认 `top_k=3`、`min_score_ratio=0.3`
- 检索器固定查询能够命中对应电路、凸透镜和电热器卡片
- context 注入、无来源回退、sources 顺序和精简字段均有 mock/fake 测试
- 当前共有 794 项单元测试，全部通过
- Stage 05 已完成普通网页问答和真实千问调用，并真实验证电热器、凸透镜 RAG 问答
- 加入相对分数过滤后再次验证电热器问题，网页来源只返回 `KB-POWER-001`
- 页面与终端验证过程中没有出现应用 traceback
- 概念验收已确认核心 RAG 数据流；Mock 测试边界和 BM25 分数含义仍需继续复习，但不是
  当前工程阻塞项
- Stage 06 的 `diagnose` 和 `explain` 已完成真实 Agent 调用验证
- Streamlit 页面已验证 HTTP 200、两组默认自动策略、旧消息兼容和清空对话
- 教师 Prompt 已加入绝对化前提检查、先勘误再回答及条件变化事实守护
- `KB-ELEC-003` 已补充额定功率定义、额定条件下实际功率相等及白炽灯丝电阻随温度
  变化的说明
- Stage 07 Function Calling 探针已确认当前 Qwen 接受白名单工具定义、返回工具调用，并
  接受匹配的 `role=tool` 消息
- Stage 07 统一 Agent 真实 E2E 使用“12 V、6 Ω 求电流”问题：Analyzer 判断
  `calculation_required=true`，Router 设置 `use_tools=true`、`use_rag=false`，
  `calculate_ohms_law` 经 Registry 本地执行得到 `2 A`，最终教师回答非空
- 上述 E2E 中 Analyzer 调用 1 次、Tool Client 调用模型 2 次，整个 Agent 共 3 次模型
  请求；普通 `answer_question` 路径未被误调用
- Stage 08 的 8 道真实 Agent 题按 Runner 固定流程各执行一次：7 题 `completed`、1 题缺图
  `blocked`、0 题失败；8 个 case_id 唯一且 answer、route、trace 等必要字段完整
- 真实评测预期路由匹配 5/8：S08-AGENT-001 和 S08-AGENT-007 比预期多启用了 RAG，
  S08-AGENT-002 被判断为 `explain` 而不是预期的 `solve`
- 真实评测共计 18 次模型请求、4 次 RAG 检索、3 次本地工具执行，平均每题 2.25 次模型请求；
  无 fallback、Retry 成功步骤、失败、空回答或单题重复工具执行
- Streamlit Trace 专项测试与完整回归通过，服务启动检查返回 HTTP 200
- Stage 09 的单图、多图、图片无文字默认问题、可选 OCR、自适应确认、缓存去重、图片上下文
  进入 RAG/Tool，以及附件与粘贴图片合并均有测试覆盖
- Stage 10 SQLite 会话、多会话管理、重启恢复、历史/状态/记忆注入、长期记忆页面交互
  均有本地测试覆盖；全量回归 581/581
- Stage 10 真实探针验证：两轮会话 hint_step 递增、记忆确认后写入且停用后不再召回；
  Streamlit HTTP 200 且无 traceback
- Stage 11.1 新增机械功率和滑轮组效率工具；Registry、参数 Schema 与计算函数均由专项测试覆盖
- Stage 11.1 支持 1～5 个工具调用的顺序执行、partial success、一次有限参数 repair 和
  Model-only fallback；已成功工具不会因 repair 或最终回答重试重复执行
- 已建立 24 条公开工具适用性评测；正式 Tool Selection 规则复测为 precision 92.3%、
  recall 80.0%、no-tool 正确率 83.3%、多工具完整命中率 100%，保留 1 次误调用作为限制
- Stage 11.2 Generation Job、Migration V3、enqueue/execute、双 Worker Task Manager、recover、
  Retry、Streamlit 状态展示、局部 Polling、多会话隔离及活动会话保护均有 Fake/Mock 测试覆盖
- Stage 11.3 Migration V4、Rolling Summary、Bridge/Recent 连续覆盖、History Retrieval、Context
  Budget、单次 Bundle 构建、三类 Projection、Current Query 优先级、后台 maintenance、Future
  终态保护、Context Trace、A/B 隔离与图片 follow-up 均有 Fake/Mock 集成测试覆盖

用户已手动验证：

- 网页计算题返回 `5 m/s`
- 网页概念题正确解释金属和木头导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

10 条知识卡片只覆盖欧姆定律与动态电路、电功率、光学、实验与易错点，不能代表完整初中
物理。BM25 仍依赖词面匹配；有效词重合与绝对/相对分数门槛只能减少低相关噪声，不能保证语义相关性或资料
正确性。当前没有向量检索、重排序、系统化召回评测或自动事实校验；真实网页验收目前只
覆盖普通问答、电热器和凸透镜等少量问题，不能代表完整 RAG 效果。

非工具题的自动模式通常进行 Analyzer 和最终回答两次模型调用；工具题通常需要 Analyzer、
工具选择和结果回传三次模型调用，会增加延迟与费用；repair 或有限 Retry 在相应场景还会
增加一次请求。Analyzer 仍可能
错误判断题型、RAG 或计算需求；Schema、Router 和 Registry 能校验结构、路由与确定性计算
参数，但不能验证复杂题目的物理建模是否正确，模型仍可能遗漏预期细节。

Stage 08 的 8 道题是工程路径小样本，未对全部回答进行系统人工物理评分，不能代表完整初中
物理能力。真实评测只有 5/8 路由与人工预期一致，说明自动模式与知识库策略仍可能偏离预期。

本地 SQLite 是单用户设计，没有登录和多用户隔离。工具链每批最多执行 5 个调用，不支持
多轮工具规划、无限 repair、任意参数猜测或基于工具结果再规划新工具；复杂多问仍可能受限。
模型偶尔可能生成字符串 `"None"` 等非法工具参数，Registry 会拒绝。
长期记忆检索目前是关键词确定性近似。未确认候选只存在 `session_state`；暂无重新启用
长期记忆的页面操作；V1 没有单独持久化 assistant 的 analysis 字段。
图片理解仍可能受清晰度、手写内容和复杂图形影响；一次最多处理 3 张图片，OCR 只有在本地
配置可用模型时才启用，不确定结果仍需人工确认。当前仅支持同批图片的有限关系融合；知识库仍
只有 10 条卡片；尚未实现向量检索、MCP、微调或通用视觉推理。

后台任务仍是单进程 `ThreadPoolExecutor` 实现，最多两个 Worker，不是分布式队列；进程退出时
遗留 running 会在下次启动变为 interrupted，需要人工 Retry。Vision/OCR 和人工确认仍在页面
线程同步完成；教师回答与 Summary maintenance 分别使用独立后台 Worker。

Context Budget 是项目内部 12000 字符安全预算，不等于 Qwen 官方最大上下文窗口；Summary 与
BM25 History Retrieval 都是有损/词面方法，不能替代原始 SQLite 消息或可靠物理知识。当前没有
Embedding、Reranker、Vector DB、跨 conversation 历史搜索或 Summary 持久任务恢复表。
