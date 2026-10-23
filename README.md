# Tests4LLMs

[![CI](https://github.com/CacinieP/Tests4LLMs/actions/workflows/ci.yml/badge.svg)](https://github.com/CacinieP/Tests4LLMs/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-%3E%3D3.9-blue)

A lightweight command-line benchmark for OpenAI-compatible LLM APIs.
面向 OpenAI 兼容 LLM API 的轻量级命令行压测工具。

`tests4llms-v0.2.py` runs burst or sustained-load tests, measures latency,
throughput and streaming TTFT, then writes machine-readable JSON and a compact
Markdown report. The benchmark itself is a single script; all configuration is
provided through CLI arguments.

`tests4llms-v0.2.py` 支持瞬发与稳态压测，统计延迟、吞吐和流式首字延迟，
并输出 JSON 原始数据与 Markdown 报告。压测程序为单脚本，全部配置通过命令行传入。

> [!CAUTION]
> This tool sends real API requests that may incur charges or trigger rate
> limits. Start with low concurrency. / 本工具会发起真实 API 请求，可能产生费用或触发限速，请从低并发开始。

## Features / 特性

- Burst and steady test modes / 瞬发与稳态两种压测模式
- Concurrency sweeps with an automatically sized connection pool / 多档并发与自动连接池
- Non-stream and SSE streaming requests with TTFT / 普通与流式请求、首字延迟
- Success rate, p50/p95 latency, req/s and token throughput / 成功率、延迟与吞吐指标
- Built-in provider presets plus manual relay configuration / 内置供应商预设与第三方中转配置
- `result.json` raw data and `report.md` summary / JSON 原始数据与 Markdown 汇总
- Environment-variable API keys; credentials are not written to reports / 密钥不写入报告

---

## Requirements and Installation / 环境与安装

- Python 3.9 or newer
- `aiohttp>=3.9`

```bash
git clone https://github.com/CacinieP/Tests4LLMs.git
cd Tests4LLMs

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Using a virtual environment is recommended but not required.
建议使用虚拟环境，但不是强制要求。

---

## Quick Start / 快速开始

### Provider preset / 供应商预设

Each preset fills in the base URL, API-key environment variable, default model,
and compatible output-token parameter.
预设会自动填写 API 地址、密钥环境变量、默认模型和输出 token 参数。

```bash
# Paste the key after the prompt; input is hidden (bash/zsh).
# 提示后粘贴密钥，输入内容不可见（bash/zsh）。
printf "STEP_API_KEY: "
read -rs STEP_API_KEY
printf "\n"
export STEP_API_KEY

python tests4llms-v0.2.py \
  --provider stepfun \
  --concurrency 1,3,5 \
  --rounds 1
```

### Manual endpoint / 手动配置

```bash
python tests4llms-v0.2.py \
  --base-url https://open.bigmodel.cn/api/paas/v4 \
  --api-key-env ZHIPU_API_KEY \
  --model glm-5 \
  --token-param max_tokens \
  --concurrency 1,3,5
```

### Steady mode / 稳态模式

```bash
python tests4llms-v0.2.py \
  --provider openai \
  --mode steady \
  --concurrency 5,10 \
  --duration 30
```

### Streaming and TTFT / 流式与首字延迟

```bash
python tests4llms-v0.2.py \
  --provider openai \
  --concurrency 1,5,10 \
  --stream
```

By default, streaming requests include `stream_options: {"include_usage": true}`.
For endpoints that reject this field, add `--no-stream-usage` to omit the entire
`stream_options` object. Missing server usage is reported as `null` in JSON and
`n/a` in Markdown; token counts are not estimated from output text.
流式请求默认请求 usage。若端点拒绝 `stream_options`，加 `--no-stream-usage`
可完全省略该字段；服务端未返回的 token 统计在 JSON 中为 `null`、Markdown 中为
`n/a`，不会根据输出文字估算。

---

## Third-party Relays / 第三方中转

Use manual configuration for an OpenAI-compatible relay. The relay must accept:

- `POST <base-url>/chat/completions`
- `Authorization: Bearer <key>`
- OpenAI Chat Completions-style request and response JSON

测试第三方中转时请使用手动参数。中转必须兼容 OpenAI Chat Completions
请求与响应格式，并支持 Bearer 鉴权。

```bash
# Paste the key, then press Enter. Input is hidden (bash/zsh).
# 粘贴密钥后按回车，输入内容不可见（bash/zsh）。
printf "RELAY_API_KEY: "
read -rs RELAY_API_KEY
printf "\n"
export RELAY_API_KEY

python tests4llms-v0.2.py \
  --base-url https://relay.example.com/v1 \
  --api-key-env RELAY_API_KEY \
  --model relay-model-id \
  --token-param max_tokens \
  --concurrency 1,3,5 \
  --rounds 1 \
  --max-tokens 100 \
  --output-dir results/relay
```

Parameter mapping / 参数对应关系：

| Relay documentation / 中转文档 | Tests4LLMs parameter / 参数 |
|----------------------------------|-----------------------------|
| Base URL such as `https://relay.example.com/v1` | `--base-url https://relay.example.com/v1` |
| Full endpoint such as `https://relay.example.com/v1/chat/completions` | Remove `/chat/completions`; use `--base-url https://relay.example.com/v1` / 去掉末尾路径 |
| API key / 密钥 | Put it in an environment variable, then use `--api-key-env RELAY_API_KEY` / 写入环境变量 |
| Model ID / 模型 ID | Use the relay's exact model name with `--model` / 使用中转实际模型名 |
| Output-token field / 输出 token 字段 | Start with `--token-param max_tokens`; switch to `max_completion_tokens` if required |

If available, query the relay's models endpoint before testing:
如果中转提供模型列表接口，建议先查询实际模型 ID：

```bash
export RELAY_BASE_URL="https://relay.example.com/v1"
curl -sS "$RELAY_BASE_URL/models" \
  -H "Authorization: Bearer $RELAY_API_KEY"
```

Useful troubleshooting / 常见排查：

- `404 model_invalid`: the relay does not recognize the `--model` value;
  query `/models` or check its documentation. / 模型名不正确，查询模型列表。
- `400` mentioning `max_tokens`: retry with
  `--token-param max_completion_tokens`. / 切换输出 token 字段。
- `429 rate_limited`: reduce `--concurrency`, `--rounds`, or steady-mode
  duration. Start with `1,3,5` and one round. / 降低并发、轮次或测试时长。
- Use `--stream` when the relay supports SSE streaming. If it rejects
  `stream_options`, add `--no-stream-usage`; usage-dependent metrics will be
  unavailable unless the server still returns usage. / 中转支持 SSE 即可启用流式；
  若拒绝 `stream_options`，加 `--no-stream-usage`，缺少 usage 时相关指标不可用。
- Relays requiring `x-api-key`, custom signatures, or other non-Bearer
  headers need a code change; the current CLI does not configure arbitrary
  headers. / 当前版本仅支持 Bearer 鉴权，不支持自定义请求头。

---

## Test Modes / 测试模式

### Burst / 瞬发模式 (`--mode burst`, default)

Fire `concurrency` requests **simultaneously**, repeat `--rounds` times.
同时发射 `concurrency` 个请求，重复 `--rounds` 轮。

- Measures: latency distribution, per-request throughput / 测量：延迟分布、单请求吞吐
- Best for: latency profiling, p50/p95 analysis / 适合：延迟分析、p50/p95 统计

### Steady / 稳态模式 (`--mode steady`)

`concurrency` workers dispatch requests during the `--duration`-second window,
then wait for requests already in flight to finish. The actual elapsed time can
exceed this window; req/s and token throughput use that full elapsed time,
including the final drain.
`concurrency` 个 worker 在 `--duration` 秒的发请求窗口内持续请求，窗口结束后等待
在途请求完成。实际耗时可能更长；req/s 与 token 吞吐的分母均为包含收尾等待的实际耗时。

- Measures: real sustained throughput (req/s, completion tok/s) / 测量：真实持续吞吐
- Best for: capacity planning, rate-limit probing / 适合：容量规划、限速探测

---

## CLI Parameters / 命令行参数

| Parameter | Default | Description / 说明 |
|-----------|---------|---------------------|
| `--provider` | — | Provider preset name (see table below) / 供应商预设名（见下表） |
| `--base-url` | required* | API prefix before `/chat/completions` / `/chat/completions` 之前的 API 地址 |
| `--api-key` | — | Plaintext API key; `--api-key-env` is safer / 明文密钥，不建议使用 |
| `--api-key-env` | — | Environment variable containing the API key / 保存密钥的环境变量名 |
| `--model` | (required*) | Model name / 模型名称 |
| `--mode` | `burst` | `burst` or `steady` / 瞬发或稳态模式 |
| `--concurrency` | `1,5,10,20` | Comma-separated concurrency levels / 并发级别 |
| `--rounds` | `1` | Burst: repeat count per level; ignored in steady mode / 瞬发模式每级轮次 |
| `--duration` | `30` | Steady: dispatch window in seconds per level; ignored in burst mode / 稳态模式每级发请求窗口（秒） |
| `--max-tokens` | `100` | Max tokens per request / 单次最大 token 数 |
| `--token-param` | preset-specific* | `max_tokens` or `max_completion_tokens`; manual default is `max_tokens` / 输出 token 字段 |
| `--stream` | off | Enable streaming, measures TTFT / 流式 + 首字延迟 |
| `--no-stream-usage` | off | Omit `stream_options` in streaming requests / 流式请求不发送 `stream_options` |
| `--prompt` | `Count from 1 to 10.` | Test prompt / 测试提示词 |
| `--timeout` | `120` | Request timeout in seconds / 超时（秒） |
| `--warmup` | `2` | Sequential warmup requests before all concurrency levels; `0` disables / 所有并发档之前顺序预热的请求数，`0` 禁用 |
| `--output-dir` | `.` | Directory for `result.json` and `report.md` / 结果目录 |

\* `--base-url` and `--model` are required unless `--provider` supplies them.
Provider presets also choose `--token-param`.

---

## Metrics / 指标

| Metric | Description / 说明 |
|--------|---------------------|
| `success_rate` | Successful requests / total (%) / 成功率 |
| `error_rate` | Failed requests / total (%) / 错误率 |
| `avg_latency_ms` | Average latency of successful requests; non-stream latency includes reading the full response body / 平均延迟；非流式延迟包含完整响应体读取 |
| `p50_latency_ms` | Median latency / 中位延迟 |
| `p95_latency_ms` | 95th percentile latency / P95 延迟 |
| `ttft_avg_ms` | Average time to first `content`, `text`, or `reasoning_content` token (stream only) / 首个正文或推理 token 延迟（仅流式） |
| `req_per_sec` | Successful requests / actual elapsed time (burst: across all rounds; steady: includes in-flight requests finishing after the dispatch window) / 成功请求数除以实际耗时，稳态含窗口结束后的收尾等待 |
| `peak_req_per_sec` | Burst only: throughput of the fastest single round / 瞬发模式独有：最快一轮的吞吐 |
| `completion_tok_per_sec` | Total completion tokens / elapsed time / 系统级输出吞吐 |
| `per_request_tok_per_sec` | Per-request: completion_tokens / latency / 单请求输出吞吐 |
| `elapsed_s` | Actual elapsed seconds per concurrency level, saved in JSON and shown as `Elapsed (s)` in the report / 每个并发档的实际耗时，写入 JSON 并显示于报告 |

Token counts and token throughput require server-provided usage. Missing usage
is `null` in JSON and `n/a` in Markdown, rather than zero or an estimate.
token 数与吞吐依赖服务端 usage；缺失时用 JSON `null` / Markdown `n/a` 表示，
不记作零，也不估算。

Non-stream latency is recorded after the full response body is read, including
for HTTP error responses. Streaming TTFT includes reasoning output, so it may
precede the first user-visible answer text.
非流式延迟在完整响应体读取后记录，HTTP 错误响应也采用同一口径。流式 TTFT 包含推理
输出，因此可能早于首个可见答案文字。

---

## Output / 输出

| File / 文件 | Content / 内容 |
|-------------|----------------|
| `result.json` | Config, per-level summaries with all metrics, raw per-request data / 配置、各级指标汇总、逐条原始数据 |
| `report.md` | 3 sections: Configuration, Overview Table, Conclusion / 三段：配置、总览表、结论 |

### report.md Sections / 报告章节

- **Configuration** — provider (`manual` without a preset), model, API, mode,
  date, prompt (with escaped line breaks), warmup count, timeout, Python and
  aiohttp versions, and request parameters. Steady duration is labeled
  **Dispatch Window**. / 供应商（未用预设时为 `manual`）、模型、API、模式、日期、
  提示词（转义换行）、预热数、超时、Python/aiohttp 版本和请求参数；稳态时长标为发请求窗口。
- **Overview** — one row per concurrency level with summary metrics and actual
  elapsed seconds / 每行一个并发级别，显示汇总指标与实际耗时
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

Presets are snapshots of the values in this script; they are not a live model
catalog. Check your service's current documentation and override parameters
without editing the script. / 预设只是脚本内的配置快照，并非实时模型目录；请按服务实际
文档通过命令行覆盖参数。

The `kimi` preset points to the Moonshot Open Platform URL shown above. Other
Kimi services, including Kimi Code, need their own endpoint, model and
credentials configured manually; do not assume credentials are interchangeable.
`kimi` 预设指向上表的 Moonshot 开放平台地址。Kimi Code 等其他 Kimi 服务应按各自的
地址、模型与凭证手动配置，不能假定不同服务的凭证通用。

The Anthropic preset uses its
[official OpenAI SDK compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk).
Provider presets also choose a compatible output-token field: OpenAI uses
`max_completion_tokens`; the other presets use `max_tokens`. Manual endpoints
default to `max_tokens`.

---

## Benchmarking Notes / 压测说明

- The benchmark measures the complete request path: local network, relay,
  provider queue and model generation. A relay result is not a model-only
  measurement. / 结果包含本地网络、中转、供应商排队和模型生成的完整链路。
- Burst results are more stable with multiple rounds; steady results are more
  representative with a longer duration. Increase load gradually. /
  多轮瞬发和更长稳态测试更有代表性，请逐步增加负载。
- Warmup runs once before the concurrency sweep: `--warmup` requests are sent
  sequentially with the measured prompt, max-token limit, token field and
  streaming options. Warmup requests are excluded from measurements, and their
  successful count is displayed. Set `--warmup 0` to disable them. Using the
  full request shape can cost more than the former short warmup. / 所有并发档之前
  只执行一次预热，顺序发出指定次数，使用与测量相同的提示词、token 上限、token 字段和
  流式选项。预热不计入结果，会显示成功数；`--warmup 0` 可关闭。完整请求预热可能比
  原来的短预热产生更多费用。
- The same prompt is reused, so provider-side caching may affect latency and
  billing. Warmup does not guarantee a cache hit, and rounds do not reliably
  identify cold versus warm requests. Even with `--rounds 1`, later concurrency
  levels may reuse a cache populated by earlier requests. / 所有请求复用同一提示词，
  供应商缓存可能影响延迟和计费；预热不保证命中缓存，轮次也不能可靠区分冷热请求。
  即使 `--rounds 1`，后续并发档也可能复用先前请求填充的缓存。
- Successful HTTP transport is not enough: responses without completion choices
  and SSE streams that end before a completion marker are counted as failures.
  Reaching the model's output-token limit is not classified as a transport
  failure. / 没有 completion choices 的响应与在结束标记前断开的 SSE 流会计为失败；
  达到模型输出 token 上限不属于传输失败。
- Do not commit API keys or paste them into `--api-key` in shared shell
  sessions. Prefer `--api-key-env`. / 请勿提交密钥，优先使用环境变量。

---

## Development / 开发

The test suite uses a local `aiohttp` server and does not call paid APIs.
测试套件使用本地模拟服务，不会调用付费 API。

```bash
python -m py_compile \
  tests4llms-v0.2.py \
  legacy/tests4llms-v0.1.py \
  tests/test_tests4llms.py

python -m unittest discover -s tests -v
```

CI also runs CLI smoke checks and flake8 fatal-error checks.
CI 还会执行 CLI 冒烟测试与 flake8 严重错误检查。

---

## Legacy Version / 旧版

`legacy/tests4llms-v0.1.py` is deprecated and retained only for historical
reference. It has hardcoded provider configuration and should not be used for
new benchmarks. / v0.1 仅作历史参考，新压测请使用 `tests4llms-v0.2.py`。

## Changelog / 更新日志

See [CHANGELOG.md](CHANGELOG.md) for release history and unreleased changes.
版本历史与尚未发布的改动见 [CHANGELOG.md](CHANGELOG.md)。

## License / 许可证

MIT
