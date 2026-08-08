# Stage 11 工具适用性专项评测

本评测只发送带 tools 的选择请求；不执行本地工具，也不生成最终教学回答。

| 策略 | Precision | Recall | No-tool 正确率 | 单工具命中 | 多工具完整命中 | 明显错误调用 | 请求数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| required_baseline | 55.6% | 100.0% | 50.0% | 100.0% | 100.0% | 12 | 24 |
| auto_current | 85.7% | 40.0% | 70.8% | 66.7% | 0.0% | 1 | 24 |
| auto_explicit | 100.0% | 73.3% | 83.3% | 55.6% | 100.0% | 0 | 24 |
| auto_formal | 92.3% | 80.0% | 83.3% | 66.7% | 100.0% | 1 | 24 |

## 失败类型

- **required_baseline**：{'should_not_call_but_did': 12}
- **auto_current**：{'should_call_but_did_not': 6, 'should_not_call_but_did': 1}
- **auto_explicit**：{'should_call_but_did_not': 4}
- **auto_formal**：{'should_call_but_did_not': 3, 'should_not_call_but_did': 1}

## 推荐

- 正式 Tool Client 使用 **auto_formal**：它复用生产中的同一条通用选择规则，本次结果为 precision 92.3%、recall 80.0%、no-tool 正确率 83.3%，明显错误调用 1。
- 候选实验中的 **auto_explicit** 仍取得最高 precision，但 `auto_formal` 额外强调“即使可心算也应使用有价值的确定性工具”，召回有所提高，同时出现 1 次误调用。该差异应记录为模型选择权衡，本轮不继续针对评测题调提示词。

## 判读边界

- `should_call_but_did_not`：已有直接、条件完整的白名单工具但模型未调用。
- `should_not_call_but_did`：概念、实验或当前工具未覆盖的计算题被调用工具。
- `wrong_tool`：调用了不在该案例期望列表内的工具。
- `multi_tool_missing`：多项独立计算未完整选择所需工具。
