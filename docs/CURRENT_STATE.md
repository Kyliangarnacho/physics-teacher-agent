# 当前状态

## Stage 05 状态

“初中物理教师 Agent + 阿里云百炼千问 API”当前使用 `qwen3.7-flash`，提示词版本仍为
冻结的 `teacher_v3_personal_humor`。Stage 05 已建立最小本地 RAG 流程，普通问答默认
保持不变，用户可在网页中选择是否使用知识库。

当前个人风格以讲解逻辑、条件分类、因果链和纠错方式为核心。幽默仅在语境自然匹配时
偶尔出现，不要求每题都有。

## 当前结构

- `app.py`：Streamlit 页面入口，负责普通/RAG 路径选择、聊天历史和来源重渲染
- `main.py`：固定平均速度题的终端回归入口
- `src/config.py`：使用 `python-dotenv` 加载并校验千问配置
- `src/prompts.py`：定义 `PROMPT_VERSION = "teacher_v3_personal_humor"` 和教师
  system prompt
- `src/model_client.py`：构造普通消息或注入参考资料 context，并调用千问
- `knowledge/physics_notes_v1.jsonl`：10 条初中物理知识卡片
- `src/retriever.py`：使用 `jieba` 和 `rank_bm25.BM25Okapi` 建立本地文本索引
- `src/rag.py`：把检索结果整理为 context，调用模型并返回回答与精简 sources
- `scripts/validate_knowledge_base.py`：校验知识库 JSONL、字段、类型和 ID 唯一性
- `evaluation/stage04_text_cases_v1.json`：15 道纯文本评测题
- `evaluation/results/`：v2 与 v3 的前 5 题 JSONL 结果
- `evaluation/reviews/`：人工评审表和新旧版本对比
- `evaluation/validate_stage04_text_cases.py`：评测数据校验入口
- `evaluation/run_stage04_text_evaluation.py`：支持 `--limit`、独立输出文件和断点续跑
- `tests/`：基于标准库 `unittest` 的评测、检索、RAG 和 context 消息测试

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

## RAG 数据流

普通模式直接调用 `answer_question(question)`，消息仍只有基础 system prompt 和当前用户
问题。

RAG 模式调用 `answer_with_rag(question, top_k=3)`：

1. `KnowledgeRetriever` 合并卡片的章节、主题、关键词和正文，使用 jieba 分词。
2. `BM25Okapi` 对卡片排序，先保留正分结果。
3. 默认以 `min_score_ratio=0.3` 删除低于最高分 30% 的结果，再限制为最多 3 条。
4. `build_context` 将卡片 ID、主题、来源和正文整理为参考资料。
5. `model_client` 在基础 system prompt 和原问题之间注入参考资料 system 消息。
6. RAG 返回回答以及只含 `id`、`topic`、`source`、`score` 的 sources。

没有正相关资料时，context 使用 `None`，仍调用普通 `answer_question`，sources 为空。

## 会话行为

`st.session_state` 只保存当前浏览器会话/标签页中的消息，页面关闭或服务重启后不会持久化。
当前页面可以显示历史消息，但每次模型请求只发送当前问题，不发送完整页面历史。

RAG 消息额外保存 `rag_enabled` 和 `sources`。网页回答下方使用折叠区域显示来源 ID、主题
和文件名，不显示知识正文；启用 RAG 但没有来源时，会提示本次按普通问答处理。

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
- 当前共有 29 项单元测试，全部通过
- Stage 05 已完成普通网页问答和真实千问调用，并真实验证电热器、凸透镜 RAG 问答
- 加入相对分数过滤后再次验证电热器问题，网页来源只返回 `KB-POWER-001`
- 页面与终端验证过程中没有出现应用 traceback
- 概念验收已确认核心 RAG 数据流；Mock 测试边界和 BM25 分数含义仍需继续复习，但不是
  当前工程阻塞项

用户已手动验证：

- 网页计算题返回 `5 m/s`
- 网页概念题正确解释金属和木头导热差异
- 当前页面能够显示两轮聊天记录

## 当前限制

10 条知识卡片只覆盖欧姆定律与动态电路、电功率、光学、实验与易错点，不能代表完整初中
物理。BM25 仍依赖词面匹配；相对分数阈值只能减少低相关噪声，不能保证语义相关性或资料
正确性。当前没有向量检索、重排序、系统化召回评测或自动事实校验；真实网页验收目前只
覆盖普通问答、电热器和凸透镜等少量问题，不能代表完整 RAG 效果。

模型每次只接收当前问题，没有多轮记忆。尚未实现图片、OCR、数据库、工具调用、MCP、
微调和真正的多轮上下文。
