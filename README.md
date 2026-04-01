# Tests4LLMs

LLM API benchmark tool for OpenAI-compatible endpoints.
面向 OpenAI 兼容 API 的 LLM 压测工具。

Single script, zero config files, all parameters via CLI.
单脚本、零配置文件，所有参数通过命令行传入。

---

## Quick Start / 快速开始

```bash
pip install aiohttp

python tests4llms-v0.2.py \
  --base-url https://open.bigmodel.cn/api/paas/v4 \
  --api-key-env ZHIPU_API_KEY \
  --model glm-4-flash \
  --concurrency 1,5,10,20 \
  --rounds 10 \
  --max-tokens 100
```

Streaming (with TTFT measurement) / 流式模式（含首字延迟测量）：

```bash
python tests4llms-v0.2.py \
  --base-url https://api.openai.com/v1 \
  --api-key sk-xxx \
  --model gpt-4o \
  --concurrency 1,5,10 \
  --rounds 10 \
  --stream
```

---

## CLI Parameters / 命令行参数

| Parameter | Default | Description / 说明 |
|-----------|---------|---------------------|
| `--base-url` | (required) | API base URL, OpenAI-compatible / API 地址（OpenAI 兼容格式） |
| `--api-key` | | API key string / API 密钥明文 |
| `--api-key-env` | | Env var name holding the API key / 存放密钥的环境变量名 |
| `--model` | (required) | Model name / 模型名称 |
| `--concurrency` | `1,5,10,20` | Comma-separated concurrency levels / 逗号分隔的并发级别 |
| `--rounds` | `10` | Requests per concurrency level / 每级并发请求数 |
| `--max-tokens` | `100` | Max tokens per request / 单次请求最大 token 数 |
| `--stream` | off | Enable streaming, measures TTFT / 启用流式输出，测量首字延迟 |
| `--prompt` | `Count from 1 to 10.` | Test prompt / 测试提示词 |
| `--timeout` | `120` | Request timeout in seconds / 请求超时（秒） |
| `--warmup` | `2` | Warmup requests before benchmark / 预热请求数 |
| `--output-dir` | `.` | Output directory / 输出目录 |

---

## Output / 输出

| File / 文件 | Content / 内容 |
|-------------|----------------|
| `result.json` | Raw details: every request's latency, tokens, status, errors / 原始明细：每条请求的延迟、token 数、状态码、错误信息 |
| `report.md` | Summary: config, concurrency table, stats, conclusion / 汇总报告：配置信息、并发对比表、统计数据、结论 |

### report.md Sections / 报告章节

- **Configuration** — model, API, date, test parameters / 模型、API、日期、测试参数
- **Concurrency Comparison** — success rate, avg latency, P50/P95, TTFT, output tok/s / 成功率、平均延迟、P50/P95、首字延迟、输出 tokens/s
- **Per-Level Details** — min/avg/p50/p95/max latency, TTFT distribution, throughput, token usage / 各级延迟分布、首字延迟、吞吐量、token 用量
- **Conclusion** — auto-generated: best latency level, throughput scaling, error warnings / 自动生成：最佳延迟级别、吞吐量扩展比、错误警告

---

## Supported Providers / 支持的供应商

Any OpenAI-compatible API, e.g.:
所有兼容 OpenAI 格式的 API，例如：

- Zhipu AI / 智谱 (`https://open.bigmodel.cn/api/paas/v4`)
- OpenAI (`https://api.openai.com/v1`)
- Anthropic via OpenAI compat / Anthropic OpenAI 兼容接口 (`https://api.anthropic.com/v1/`)
- Google Gemini (`https://generativelanguage.googleapis.com/v1beta/openai/`)
- MiniMax, Kimi/Moonshot / 月之暗面, StepFun / 阶跃星辰, DeepSeek / 深度求索, etc.

---

## Changelog / 更新日志

### v0.2 (`tests4llms-v0.2.py`)

| EN | 中文 |
|----|------|
| Single script, zero source editing — all config via CLI args | 单脚本、免改源码 — 所有配置走命令行参数 |
| Markdown report (`report.md`): concurrency table, per-level details, auto conclusion | Markdown 报告：并发对比表、逐级详情、自动结论 |
| Streaming + TTFT: `--stream` enables SSE parsing, measures time-to-first-token | 流式 + 首字延迟：`--stream` 启用 SSE 解析，测量 TTFT |
| Output tokens/s: per-request throughput (completion tokens / latency) | 输出 tokens/s：单请求吞吐量（completion tokens / 延迟） |
| Structured JSON (`result.json`): config, summaries, raw per-request data | 结构化 JSON：配置、汇总、逐条原始数据 |
| Configurable warmup to establish connections | 可配置预热请求数，预先建立连接 |

### v0.1 (`tests4llms-v0.1.py`)

| EN | 中文 |
|----|------|
| Initial version: async benchmark with connection pooling | 初始版本：asyncio + 连接池压测 |
| Latency test, concurrency sweep, RPM/TPM window test | 延迟测试、并发扫描、RPM/TPM 窗口测试 |
| Provider config hardcoded in source (BASE_URL / API_KEY_ENV / DEFAULT_MODEL) | 供应商配置硬编码在源码中 |
| JSON output only | 仅 JSON 输出 |

---

## License / 许可证

MIT
