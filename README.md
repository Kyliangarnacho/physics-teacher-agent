# Physics Teacher Agent

## 第一阶段定位

本项目第一阶段为“初中物理教师 Agent + 阿里云百炼千问 API”。项目通过 OpenAI
兼容接口调用千问模型，提供简单、清晰的初中物理文本问答。

## 环境与运行

- Python 3.12 虚拟环境：`.venv`
- 当前入口：`python main.py`
- 依赖：`openai`、`python-dotenv`
- 配置：将 API Key 保存在本地 `.env` 中；`.env` 已被 Git 忽略且未提交

运行前需要在 `.env` 中配置 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL` 和
`QWEN_MODEL`。可参考 `.env.example`，不要提交真实 API Key。

## 已完成功能

- 使用 `python-dotenv` 加载本地配置
- 检查必需配置并给出中文缺失提示
- 使用 OpenAI 客户端的 `chat.completions` 调用阿里云百炼千问 API
- 设置初中物理教师角色并输出模型返回的文本回答
- 对认证、网络、限流和其他 API 错误给出简短中文提示

## 第一阶段测试

1. 平均速度计算题：正确得到 `50 m ÷ 10 s = 5 m/s`，并给出计算过程。
2. 概念题：选择并测试“为什么冬天金属摸起来比木头更凉，但它们实际温度可能相同？”，
   正确说明金属导热更快、手部热量流失更快。
3. 配置缺失测试：临时让程序读取不到 `QWEN_MODEL`，程序清楚提示缺失配置；恢复正常
   配置后再次调用成功。

## 当前限制

第一阶段仅支持文本问答，暂不包含网页、图片输入、数据库、RAG 或 Agent 框架。
