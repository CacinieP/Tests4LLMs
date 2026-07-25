# Tests4LLMs

LLM API benchmark tool for OpenAI-compatible endpoints.
面向 OpenAI 兼容 API 的 LLM 压测工具。

Single script, zero config files, all parameters via CLI.
单脚本、零配置文件，所有参数通过命令行传入。

**Requires Python >= 3.9.** `aiohttp` is the only runtime dependency (`pip install -r requirements.txt`).
See [CHANGELOG.md](CHANGELOG.md) for the full version history.

> `legacy/tests4llms-v0.1.py` is **deprecated** and kept for reference only — it has hardcoded provider config and an unused `keepalive_connections` parameter. Use `tests4llms-v0.2.py`.

---

## Quick Start / 快速开始

```bash
pip install aiohttp

# Provider preset (recommended)
python tests4llms-v0.2.py \
  --provider zhipu \
  --concurrency 1,5,10,20 \
  --rounds 3

# Manual config
python tests4llms-v0.2.py \
  --base-url https://open.bigmodel.cn/api/paas/v4 \
  --api-key-env ZHIPU_API_KEY \
  --model glm-5 \
  --concurrency 1,5,10,20 \
  --rounds 3
```

Steady mode — sustained throughput test / 稳态吞吐测试：

```bash
python tests4llms-v0.2.py \
  --provider openai \
  --mode steady \
  --concurrency 5,10 \
  --duration 30
```

Streaming (with TTFT measurement) / 流式模式（含首字延迟测量）：

```bash
python tests4llms-v0.2.py \
  --provider openai \
  --concurrency 1,5,10 \
  --stream
```

---

## Test Modes / 测试模式

### Burst / 瞬发模式 (`--mode burst`, default)

Fire `concurrency` requests **simultaneously**, repeat `--rounds` times.
同时发射 `concurrency` 个请求，重复 `--rounds` 轮。

- Measures: latency distribution, per-request throughput / 测量：延迟分布、单请求吞吐
- Best for: latency profiling, p50/p95 analysis / 适合：延迟分析、p50/p95 统计

### Steady / 稳态模式 (`--mode steady`)

`concurrency` workers sustain load for `--duration` seconds.
`concurrency` 个 worker 持续发请求 `--duration` 秒。

- Measures: real sustained throughput (req/s, completion tok/s) / 测量：真实持续吞吐
- Best for: capacity planning, rate-limit probing / 适合：容量规划、限速探测

---

## CLI Parameters / 命令行参数

| Parameter | Default | Description / 说明 |
|-----------|---------|---------------------|
| `--provider` | | Provider preset name (see table below) / 供应商预设名（见下表） |
| `--base-url` | (required*) | API base URL, OpenAI-compatible / API 地址 |
| `--api-key` | | API key string / API 密钥明文 |
| `--api-key-env` | | Env var name holding the API key / 环境变量名 |
| `--model` | (required*) | Model name / 模型名称 |
| `--mode` | `burst` | `burst` or `steady` / 瞬发或稳态模式 |
| `--concurrency` | `1,5,10,20` | Comma-separated concurrency levels / 并发级别 |
| `--rounds` | `1` | Burst: repeat count per level / 瞬发模式每级重复次数 |
| `--duration` | `30` | Steady: test duration in seconds / 稳态模式时长（秒） |
| `--max-tokens` | `100` | Max tokens per request / 单次最大 token 数 |
| `--token-param` | provider-specific | Output-token request field: `max_tokens` or `max_completion_tokens` / 输出 token 请求字段 |
| `--stream` | off | Enable streaming, measures TTFT / 流式 + 首字延迟 |
| `--prompt` | `Count from 1 to 10.` | Test prompt / 测试提示词 |
| `--timeout` | `120` | Request timeout in seconds / 超时（秒） |
| `--warmup` | `2` | Warmup requests / 预热请求数 |
| `--output-dir` | `.` | Output directory / 输出目录 |

\* Required unless `--provider` is used.

---

## Metrics / 指标

| Metric | Description / 说明 |
|--------|---------------------|
| `success_rate` | Successful requests / total (%) / 成功率 |
| `error_rate` | Failed requests / total (%) / 错误率 |
| `avg_latency_ms` | Average latency of successful requests / 平均延迟 |
| `p50_latency_ms` | Median latency / 中位延迟 |
| `p95_latency_ms` | 95th percentile latency / P95 延迟 |
| `ttft_avg_ms` | Average time-to-first-token (stream only) / 首字延迟（仅流式） |
| `req_per_sec` | Average successful requests per second (burst: across all rounds; steady: sustained over the window) / 平均每秒成功请求数 |
| `peak_req_per_sec` | Burst only: throughput of the fastest single round / 瞬发模式独有：最快一轮的吞吐 |
| `completion_tok_per_sec` | Total completion tokens / elapsed time / 系统级输出吞吐 |
| `per_request_tok_per_sec` | Per-request: completion_tokens / latency / 单请求输出吞吐 |

---

## Output / 输出

| File / 文件 | Content / 内容 |
|-------------|----------------|
| `result.json` | Config, per-level summaries with all metrics, raw per-request data / 配置、各级指标汇总、逐条原始数据 |
| `report.md` | 3 sections: Configuration, Overview Table, Conclusion / 三段：配置、总览表、结论 |

### report.md Sections / 报告章节

- **Configuration** — model, API, mode, date, params / 模型、API、模式、日期、参数
- **Overview** — one row per concurrency level with all metrics / 每行一个并发级别，包含全部指标
- **Conclusion** — auto-generated: best latency level, best throughput level, success rate inflection point / 自动生成：最佳低延迟档、最佳吞吐档、成功率拐点

---

## Supported Providers / 支持的供应商

Use `--provider <name>` for quick setup, or manually specify `--base-url --api-key-env --model`.
使用 `--provider <name>` 快速配置，或手动指定三个参数。

| Provider / 供应商 | --provider | --base-url | --api-key-env | --model |
|-------------------|------------|------------|---------------|---------|
| 智谱 Zhipu AI | `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | `glm-5` |
| MiniMax | `minimax` | `https://api.minimax.io/v1` | `MINIMAX_API_KEY` | `MiniMax-M2.7` |
| Kimi / 月之暗面 Moonshot | `kimi` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | `kimi-k2.5` |
| StepFun / 阶跃星辰 | `stepfun` | `https://api.stepfun.com/v1` | `STEP_API_KEY` | `step-3.7-flash` |
| Google Gemini | `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | `gemini-3.1-pro-preview` |
| OpenAI | `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `gpt-5.4` |
| Anthropic | `anthropic` | `https://api.anthropic.com/v1/` | `ANTHROPIC_API_KEY` | `claude-opus-5` |

> **Anthropic:** this preset uses Anthropic's official OpenAI SDK compatibility layer. It is suitable for testing and evaluation, including streaming and usage reporting.

Provider presets also choose a compatible output-token field. OpenAI uses
`max_completion_tokens`; the other presets use `max_tokens`. Manual endpoints
default to `max_tokens` and can override it with `--token-param`.

---

## Changelog / 更新日志

> Full history: [CHANGELOG.md](CHANGELOG.md)

### Unreleased (`tests4llms-v0.2.py`) — current / 当前版本

Correctness and coverage follow-up to v0.2.1.

| EN | 中文 |
|----|------|
| Empty or truncated streams are failures; `[DONE]` ends a request immediately | 空流或截断流会判定失败；收到 `[DONE]` 立即结束请求 |
| Invalid concurrency, duration, rounds, timeout, and token limits fail fast | 非法并发、时长、轮次、超时和 token 上限会立即报错 |
| The connection pool automatically matches the highest requested concurrency | 连接池会自动匹配最高并发级别 |
| Provider-specific output-token parameters, including OpenAI `max_completion_tokens` | 按供应商选择输出 token 参数，包括 OpenAI 的 `max_completion_tokens` |
| Restored Anthropic through its official OpenAI SDK compatibility layer | 通过 Anthropic 官方 OpenAI SDK 兼容层恢复 Anthropic 预设 |
| Behavioral tests now cover CLI validation, streaming, payloads, and pool sizing | 新增行为测试，覆盖参数校验、流式响应、请求载荷和连接池 |
| Legacy v0.1 no longer logs any API-key prefix | 旧版 v0.1 不再输出 API 密钥前缀 |

Earlier versions are documented in [CHANGELOG.md](CHANGELOG.md).


---

## License / 许可证

MIT
