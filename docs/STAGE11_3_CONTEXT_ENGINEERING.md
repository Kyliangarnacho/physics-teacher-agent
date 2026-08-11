# Stage 11.3 Context Engineering

## 最终架构

```text
Messages
→ Rolling Summary / Unsummarized Bridge / Recent 3 turns
→ conversation-scoped History Retrieval
→ Context Manager + deterministic character Budget
→ one ContextBundle snapshot per Generation
→ Analyzer / Tool / Final Projection
→ Agent
```

- `Summary + Bridge + Recent` 连续覆盖完整历史；Summary 未覆盖的旧轮次始终保留在 Bridge。
- Retrieved 只搜索当前 conversation 中 Summary-covered 的完整旧轮次，排除 Recent、Bridge、
  当前/孤立 user 和非 completed Job；旧 assistant 不是权威物理知识。
- ContextBundle 每个 Generation 只构造一次。Analyzer/Final 使用宽视图；Tool 使用窄视图且不携带
  Learning Memory。Current Query 是本轮唯一执行目标。
- Budget 是 12000 字符的项目内部安全预算，不代表模型官方 Context Window。裁剪顺序为 Retrieved、
  Learning Memory、Rolling Summary；ConversationState、Recent 和 Bridge 优先保护并保留完整轮次。
- Trace 只保存 Summary revision、窗口数量、预算和 Projection 字符数等统计，不保存上下文原文。

后台维护：

```text
Generation completed
→ O(1) maintenance submit
→ ContextMaintenance Worker
→ persisted stale check
→ Rolling Summary refresh
```

不建立 `summary_jobs`。Summary 失败不会改变旧 Summary、covered boundary、Bridge 或已完成回答；
Generation Future 异常会落成 terminal failure，failed/interrupted 可复用原 Job Retry。

## Learning Memory 边界

Learning Memory 沿用 Stage 10 合同：普通一次错误不能直接成为 weakness，明确错误规律才可作为
misconception，preference 需要用户明确表达；候选必须由用户确认后才持久化。Context Manager 只检索、
格式化 confirmed/active Memory，失败时降级为空，不执行提取。Conversation Summary 维护本会话事实
连续性，Learning Memory 保存用户确认的稳定学习信息，两者严格分离。

## 关键调试结论

- Summary Client 必须按 `load_qwen_config()` 的 `(api_key, base_url, model)` 三元组读取，不能当对象属性。
- 只清理 scheduled 而不读取后台 Future 异常，会让 Generation Job 残留 `running`；Future callback
  必须保证终态，进程重启遗留 running 仍由 recover 标记为 interrupted。
- BM25 只有 `max_score * ratio` 会“矮子里拔将军”；需要先检查最高分的绝对相关性。
- `a/b/的/大/小/解释/受到` 等低信息词会制造伪高分；History Retrieval 还需要有效词过滤和主题重合门槛。
- stale scan 曾在 Generation completion 中同步执行；完成路径现在只 enqueue，扫描和 Summary API
  全部位于独立 Worker。
- Trace 数据曾已写入 SQLite，但 UI 展示不清；当前 renderer 固定展示 Context metadata、budget 和
  projections，且不回显上下文内容。

## 验证范围

自动测试覆盖 Summary/Bridge/Recent 连续性、revision 滚动更新、History Retrieval 空结果与隔离、
单次 Bundle 构建、Projection/Current Query 优先级、Generation failure/Retry、后台 maintenance、
Trace 持久化、A/B conversation、Learning Memory 安全降级及图片 follow-up。测试使用 Fake/Mock，
不调用真实 API。
