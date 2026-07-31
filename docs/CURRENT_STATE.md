# 当前状态

## Stage 06 状态

“初中物理教师 Agent + 阿里云百炼千问 API”当前使用 `qwen3.7-flash`，提示词版本仍为
`teacher_v3_personal_humor`。Stage 06 已在 Stage 05 最小本地 RAG 上增加结构化
Question Analyzer、Router、四种教学模式、三态 RAG 策略、统一 Agent 编排和网页决策
展示。

当前个人风格以讲解逻辑、条件分类、因果链和纠错方式为核心。幽默仅在语境自然匹配时
偶尔出现，不要求每题都有。

## 当前结构

- `app.py`：Streamlit 页面入口，统一调用 Agent，保存并重渲染回答、决策和来源
- `main.py`：固定平均速度题的终端回归入口
- `src/config.py`：使用 `python-dotenv` 加载并校验千问配置
- `src/schemas.py`：定义 `TeachingMode`、`QuestionAnalysis` 和 `RouteDecision`
- `src/analyzer.py`：执行第一次模型调用、解析并校验问题分析，失败时安全 fallback
- `src/router.py`：处理模式覆盖、RAG 策略、缺图拦截和回答许可
- `src/agent.py`：统一编排分析、路由、可选 RAG 与最终回答
- `src/prompts.py`：定义教师 Prompt、Analyzer Prompt 和 `MODE_INSTRUCTIONS`
- `src/model_client.py`：注入模式指令和可选参考资料，并调用最终回答模型
- `knowledge/physics_notes_v1.jsonl`：10 条初中物理知识卡片
- `src/retriever.py`：使用 `jieba` 和 `rank_bm25.BM25Okapi` 建立本地文本索引
- `src/rag.py`：把检索结果整理为 context，调用模型并返回回答与精简 sources
- `scripts/validate_knowledge_base.py`：校验知识库 JSONL、字段、类型和 ID 唯一性
- `evaluation/stage04_text_cases_v1.json`：15 道纯文本评测题
- `evaluation/results/`：v2 与 v3 的前 5 题 JSONL 结果
- `evaluation/reviews/`：人工评审表和新旧版本对比
- `evaluation/validate_stage04_text_cases.py`：评测数据校验入口
- `evaluation/run_stage04_text_evaluation.py`：支持 `--limit`、独立输出文件和断点续跑
- `tests/`：基于标准库 `unittest` 的 Schema、Analyzer、Router、Agent、评测、检索、
  RAG、消息和事实守护测试

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

# 校验 Stage 05 知识库
.venv\Scripts\python.exe scripts\validate_knowledge_base.py

# 运行全部本地单元测试（不调用 API）
.venv\Scripts\python.exe -m unittest discover -v
```

## Agent 数据流

```text
用户问题
→ Analyzer 第一次模型调用
→ QuestionAnalysis 校验
→ Router
→ RouteDecision
→ 可选 RAG
→ 模式指令和可选参考资料
→ 第二次模型调用
→ 页面展示答案、决策和来源
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

## 会话行为

`st.session_state` 只保存当前浏览器会话/标签页中的消息。新 Assistant 消息保存
`content`、`sources`、`analysis`、`route` 和 `analysis_fallback`，页面可折叠显示
Agent 决策和来源；旧格式消息仍能安全显示。

浏览器新会话、硬刷新导致会话重建或服务重启后，历史不会持久化。页面显示历史不等于
模型记忆；每次模型请求仍只发送当前问题，不发送完整页面历史。

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
- 当前共有 83 项单元测试，全部通过
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

用户已手动验证：

- 网页计算题返回 `5 m/s`
- 网页概念题正确解释金属和木头导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

10 条知识卡片只覆盖欧姆定律与动态电路、电功率、光学、实验与易错点，不能代表完整初中
物理。BM25 仍依赖词面匹配；相对分数阈值只能减少低相关噪声，不能保证语义相关性或资料
正确性。当前没有向量检索、重排序、系统化召回评测或自动事实校验；真实网页验收目前只
覆盖普通问答、电热器和凸透镜等少量问题，不能代表完整 RAG 效果。

自动模式通常进行 Analyzer 和最终回答两次模型调用，会增加延迟与费用。Analyzer 仍可能
发生语义分类错误；Schema 和 Router 只能校验结构和流程，不能验证最终回答的物理事实，
模型仍可能遗漏预期细节。

当前没有真正的多轮上下文记忆，页面历史不会传入模型，也不会跨浏览器新会话或服务重启
持久化。当前不支持图片上传和图片理解，`image_required` 只是缺图安全拦截。知识库仍
只有 10 条卡片，覆盖有限；尚未实现 OCR、向量检索、数据库、工具调用、MCP 或微调。
