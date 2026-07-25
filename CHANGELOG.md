# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(from v0.2.1 onward). Versions prior to v0.2.1 were not formally tagged.

## [Unreleased]

### Added
- Behavioral tests using a local `aiohttp` server. They cover argument
  validation, provider defaults, request payloads, connection-pool sizing,
  empty streams, truncated streams, `[DONE]`, and compatible streams that
  finish without a `[DONE]` marker.
- `--token-param` for endpoints that need an explicit
  `max_tokens`/`max_completion_tokens` override.

### Changed
- The connection-pool limit now matches the highest requested concurrency
  instead of silently capping every run at 256 connections.
- OpenAI requests use the current `max_completion_tokens` field. Other
  provider presets use `max_tokens`.
- The StepFun preset now defaults to the verified `step-3.7-flash` model.
- Steady-mode workers no longer sleep between requests, so the mode measures
  sustained endpoint throughput without client-side pacing.
- CI now runs the behavioral test suite and compile-checks and lints the test
  module as well as the main script.

### Fixed
- Streaming TTFT now recognizes `delta.reasoning_content`, used by reasoning
  models such as Kimi-K3, in addition to `delta.content` and `delta.text`.
- Streaming requests stop reading as soon as `[DONE]` arrives instead of
  waiting for the server to close the connection.
- HTTP 200 streams with no completion chunks, and streams that end before
  `[DONE]` or a `finish_reason`, are now failures rather than false successes.
- Non-stream HTTP 200 responses without completion choices are also failures.
- Invalid empty, non-integer, zero, or negative concurrency values fail fast.
  Burst rounds, steady duration, timeout, max tokens, warmup count, and output
  directory are also validated before the benchmark starts.
- Restored the Anthropic preset through Anthropic's official OpenAI SDK
  compatibility endpoint, using `claude-opus-5`.
- The deprecated v0.1 script no longer prints any part of the API key.

## [0.2.1] - 2026-07-25

The "repair" release: correctness fixes, streaming robustness, a real CI, and
documentation cleanup. This release temporarily removed the `anthropic`
preset because its compatibility endpoint was mistaken for a third-party
proxy; the preset is restored under Unreleased.

### Removed
- **Anthropic provider preset (temporary).** It was removed based on an
  incorrect compatibility assumption and is restored under Unreleased using
  Anthropic's official OpenAI SDK compatibility layer.
- Duplicated `PROVIDER_NAMES` constant. The `--provider` help text is now
  generated from the `PROVIDERS` dict, so the two can never drift.

### Added
- `requirements.txt` (`aiohttp>=3.9`) — the dependency CI installs.
- **`peak_req_per_sec`** metric in burst mode: throughput of the fastest
  single round. Appears in `result.json`, the `report.md` overview table
  (extra `req/s (peak)` column, burst mode only), and the auto-generated
  conclusion ("best throughput" now prefers peak).
- Smoke tests to CI: `--help` parse, unknown-provider rejection, and
  `py_compile` on both `tests4llms-v0.2.py` and `legacy/tests4llms-v0.1.py`.
- Declared **Python >= 3.9** requirement (script uses PEP 585 `list[str]`).

### Changed
- **Same fixed prompt for every request.** Previously each request's prompt
  was suffixed with `(#n)`, which (a) made per-request throughput
  comparisons unfair and (b) defeated prompt caching on providers that
  support it (OpenAI, Anthropic). All requests now send the identical
  `--prompt`.
- `req_per_sec` semantics clarified: in burst mode it is the average across
  all rounds (was implied to be "the" throughput); the new
  `peak_req_per_sec` captures the burst peak. Steady mode is unchanged.
- Streaming TTFT detection now reads both `delta.content` **and**
  `delta.text`, so gemini-style gateways register a correct first-token
  time. Chunks carrying only `delta.role` no longer trigger a false TTFT.
- `report.md` overview table is now built dynamically from the actual
  metrics present (TTFT column only in stream runs; peak column only in
  burst runs) instead of being hardcoded per mode.

### Fixed
- **CI no longer swallows failures.** Every step previously ended with
  `|| true`, so the workflow was perpetually green while installing nothing,
  running zero tests, and skipping lint. All `|| true` removed; the workflow
  now fails loudly on any real error.
- **Mid-stream errors no longer crash the batch.** A timeout or connection
  error during SSE reading used to propagate out of `_stream_request`,
  breaking `asyncio.gather` for the whole concurrency batch. It now returns
  a proper `BenchResult(success=False, ...)` carrying the `ttft_ms` measured
  so far and a descriptive error (`timeout (mid-stream)` / `stream_error`).
- **`RuntimeWarning: coroutine '__aexit__' was never awaited.** The
  streaming path manually entered/exited the request context manager and
  forgot to await `__aexit__`. Rewritten to use a clean `async with`, so the
  response is always released.

### Deprecated
- `tests4llms-v0.1.py` moved to **`legacy/tests4llms-v0.1.py`** with a
  deprecation banner. Known limitations at the time (do not use in
  production):
  hardcoded provider config (no `--provider` flag), `LLMClient.__init__`
  accepts `keepalive_connections` but never applies it, prints the first 12
  characters of the API key, and `top_errors()` can split UTF-8 multibyte
  characters mid-sequence.

## [0.2] - 2026-07-22

The rewrite: single script, dual test modes, provider presets, streaming.

### Added
- Six logical blocks: CLI, Provider Profile, Runner, Metrics, Reporter, IO.
- Dual mode: `burst` (instantaneous concurrency) + `steady` (sustained
  throughput over a duration window).
- `--provider` preset: one flag sets `base-url` / `key-env` / `model`.
- Rich metrics: latency p50/p95, TTFT, `req/s`, `completion_tok/s`,
  per-request `tok/s`.
- Streaming + TTFT: `--stream` enables SSE parsing.
- Structured `result.json` (config + summaries + raw per-request data).
- Simplified `report.md`: Configuration + Overview Table + Conclusion
  (auto-identifies best latency/throughput levels and success-rate
  inflection point).

## [0.1] - 2026-07-20

Initial version (now archived under `legacy/`).

### Added
- Asyncio + aiohttp connection-pooling benchmark client.
- Latency test, concurrency sweep, RPM/TPM window test.
- Warmup phase.
- JSON output.

### Limitations (unfixed in v0.1)
- Provider config hardcoded in source.
- JSON output only (no markdown report).

[Unreleased]: https://github.com/CacinieP/Tests4LLMs/compare/eda12ba...HEAD
[0.2.1]: https://github.com/CacinieP/Tests4LLMs/compare/03ec697...eda12ba
[0.2]: https://github.com/CacinieP/Tests4LLMs/compare/b8d3a9b...03ec697
[0.1]: https://github.com/CacinieP/Tests4LLMs/commit/b8d3a9b
