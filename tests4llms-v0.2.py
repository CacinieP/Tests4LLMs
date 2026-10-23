#!/usr/bin/env python3
"""
Tests4LLMs — LLM API Benchmark (Single Script)

Requires Python >= 3.9 (uses PEP 585 generic aliases like list[str]).

Outputs: result.json (raw details) + report.md (summary with tables)

支持的 API（OpenAI 兼容格式），使用 --provider 快捷切换或手动指定参数：

## 1. 智谱 Zhipu AI
# python tests4llms-v0.2.py --provider zhipu
# 或手动指定:
# python tests4llms-v0.2.py --base-url https://open.bigmodel.cn/api/paas/v4 --api-key-env ZHIPU_API_KEY --model glm-5

## 2. MiniMax
# python tests4llms-v0.2.py --provider minimax

## 3. Kimi（月之暗面 Moonshot）
# python tests4llms-v0.2.py --provider kimi

## 4. StepFun（阶跃星辰）
# python tests4llms-v0.2.py --provider stepfun

## 5. Google Gemini
# python tests4llms-v0.2.py --provider gemini

## 6. OpenAI
# python tests4llms-v0.2.py --provider openai

## 7. Anthropic
# python tests4llms-v0.2.py --provider anthropic

Usage:
  pip install aiohttp

  # Burst mode: 瞬时并发测试
  python tests4llms-v0.2.py \\
    --provider openai \\
    --concurrency 1,5,10,20 \\
    --rounds 3

  # Steady mode: 持续吞吐测试
  python tests4llms-v0.2.py \\
    --provider openai \\
    --mode steady \\
    --concurrency 5,10 \\
    --duration 30

  # Streaming (TTFT measurement)
  python tests4llms-v0.2.py \\
    --provider openai \\
    --concurrency 1,5,10 \\
    --stream
"""

import argparse
import asyncio
import html
import json
import math
import os
import platform
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, List, Optional

import aiohttp


# ═══════════════════════════════════════════════════════════════════
# 1. CLI — Argument Parsing
# ═══════════════════════════════════════════════════════════════════

# Providers are declared below in PROVIDERS; the help text is derived from it
# so the two never drift out of sync.


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tests4LLMs — LLM API Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tests4llms-v0.2.py --provider zhipu "
            "--concurrency 1,5,10\n"
            "  python tests4llms-v0.2.py --provider openai --mode steady "
            "--concurrency 5,10 --duration 30\n"
            "  python tests4llms-v0.2.py "
            "--base-url https://api.openai.com/v1 "
            "--api-key-env OPENAI_API_KEY --model gpt-5.4 --stream\n"
        ),
    )

    g = p.add_argument_group("API")
    g.add_argument("--base-url", default="",
                   help="API base URL (OpenAI-compatible). "
                        "Optional if --provider is used.")
    g.add_argument("--api-key", default="",
                   help="API key (or use --api-key-env)")
    g.add_argument("--api-key-env", default="",
                   help="Env var name holding the API key")
    g.add_argument("--model", default="",
                   help="Model name to benchmark. Optional if --provider is used.")
    g.add_argument("--provider", default="",
                   help=f"Provider preset: {', '.join(PROVIDERS)}")
    g.add_argument(
        "--token-param",
        choices=["max_tokens", "max_completion_tokens"],
        default="",
        help="Output-token request field. Provider presets choose a compatible "
             "default; manual configuration defaults to max_tokens.",
    )

    g = p.add_argument_group("Mode")
    g.add_argument("--mode", choices=["burst", "steady"], default="burst",
                   help="burst: fire concurrency requests at once, repeat --rounds "
                        "times. steady: sustain concurrency workers for --duration "
                        "seconds (default: burst)")
    g.add_argument("--concurrency", default="1,5,10,20",
                   help="Comma-separated concurrency levels (default: 1,5,10,20)")
    g.add_argument("--rounds", type=int, default=1,
                   help="Burst mode: repeat count per level (default: 1). "
                        "Ignored in steady mode.")
    g.add_argument("--duration", type=int, default=30,
                   help="Steady mode: test duration in seconds (default: 30). "
                        "Ignored in burst mode.")

    g = p.add_argument_group("Request")
    g.add_argument("--max-tokens", type=int, default=100,
                   help="Max tokens per request (default: 100)")
    g.add_argument("--stream", action="store_true",
                   help="Enable streaming (measures TTFT)")
    g.add_argument("--no-stream-usage", dest="stream_usage", action="store_false",
                   help="Omit stream_options for endpoints that reject it. "
                        "Token rates need usage returned by the server.")
    g.add_argument("--prompt", default="Count from 1 to 10.",
                   help="Test prompt (default: 'Count from 1 to 10.')")

    g = p.add_argument_group("Tuning")
    g.add_argument("--timeout", type=float, default=120.0,
                   help="Request timeout in seconds (default: 120)")
    g.add_argument("--warmup", type=int, default=2,
                   help="Requests using the benchmark prompt and token limit, "
                        "once before all levels (default: 2; 0 disables)")
    g.add_argument("--output-dir", default=".",
                   help="Directory for result.json & report.md (default: .)")

    return p


def resolve_api_key(args) -> str:
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "")
        if not key:
            raise SystemExit(f"ERROR: Environment variable {args.api_key_env} is not set")
        return key
    raise SystemExit("ERROR: Provide --api-key or --api-key-env")


def parse_concurrency(s: str) -> List[int]:
    try:
        levels = [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(
            "ERROR: --concurrency must be a comma-separated list of integers"
        ) from exc
    if not levels:
        raise SystemExit("ERROR: --concurrency must contain at least one level")
    invalid = sorted(set(level for level in levels if level <= 0))
    if invalid:
        values = ", ".join(str(level) for level in invalid)
        raise SystemExit(
            f"ERROR: --concurrency levels must be positive; got: {values}"
        )
    return sorted(set(levels))


# ═══════════════════════════════════════════════════════════════════
# 2. Provider Profile — Endpoint Presets
# ═══════════════════════════════════════════════════════════════════

PROVIDERS: Dict[str, Dict[str, str]] = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "ZHIPU_API_KEY",
        "model": "glm-5",
        "token_param": "max_tokens",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "key_env": "MINIMAX_API_KEY",
        "model": "MiniMax-M2.7",
        "token_param": "max_tokens",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "key_env": "MOONSHOT_API_KEY",
        "model": "kimi-k2.5",
        "token_param": "max_tokens",
    },
    "stepfun": {
        "base_url": "https://api.stepfun.com/v1",
        "key_env": "STEP_API_KEY",
        "model": "step-3.7-flash",
        "token_param": "max_tokens",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.1-pro-preview",
        "token_param": "max_tokens",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4",
        "token_param": "max_completion_tokens",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "key_env": "ANTHROPIC_API_KEY",
        "model": "claude-opus-5",
        "token_param": "max_tokens",
    },
}


def apply_provider(args):
    """Apply provider preset, letting explicit CLI args take precedence."""
    if not args.provider:
        return
    key = args.provider.lower()
    if key not in PROVIDERS:
        raise SystemExit(
            f"ERROR: Unknown provider '{args.provider}'. "
            f"Available: {', '.join(PROVIDERS.keys())}"
        )
    preset = PROVIDERS[key]
    if not args.base_url:
        args.base_url = preset["base_url"]
    if not args.api_key_env:
        args.api_key_env = preset["key_env"]
    if not args.model:
        args.model = preset["model"]
    if not args.token_param:
        args.token_param = preset["token_param"]


def validate_args(args):
    """Validate required args after provider resolution."""
    if not args.base_url:
        raise SystemExit("ERROR: --base-url is required (or use --provider)")
    if not args.model:
        raise SystemExit("ERROR: --model is required (or use --provider)")
    if not args.token_param:
        args.token_param = "max_tokens"
    if args.mode == "burst" and args.rounds <= 0:
        raise SystemExit("ERROR: --rounds must be positive in burst mode")
    if args.mode == "steady" and args.duration <= 0:
        raise SystemExit("ERROR: --duration must be positive in steady mode")
    if args.max_tokens <= 0:
        raise SystemExit("ERROR: --max-tokens must be positive")
    if args.timeout <= 0:
        raise SystemExit("ERROR: --timeout must be positive")
    if args.warmup < 0:
        raise SystemExit("ERROR: --warmup cannot be negative")
    if not str(args.output_dir).strip():
        raise SystemExit("ERROR: --output-dir cannot be empty")


# ═══════════════════════════════════════════════════════════════════
# 3. Runner — Test Execution
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    success: bool
    status_code: int
    latency_ms: float
    ttft_ms: float = 0.0
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    tokens_total: Optional[int] = None
    per_request_tok_per_sec: Optional[float] = None
    error: Optional[str] = None


class LLMClient:
    """Async LLM client with connection pooling (OpenAI-compatible)."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout_s: float = 120.0, stream: bool = False,
                 max_connections: int = 256,
                 token_param: str = "max_tokens", stream_usage: bool = True):
        if max_connections <= 0:
            raise ValueError("max_connections must be positive")
        if token_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"unsupported token parameter: {token_param}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.stream_usage = stream_usage
        self.token_param = token_param
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.connector = aiohttp.TCPConnector(
            limit=max_connections, limit_per_host=max_connections,
            keepalive_timeout=30, enable_cleanup_closed=True,
            ttl_dns_cache=300,
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=self.connector,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()

    async def chat(self, prompt: str, max_tokens: int = 100) -> BenchResult:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            self.token_param: max_tokens,
            "stream": self.stream,
        }
        start = time.perf_counter()
        try:
            if self.stream:
                return await self._stream_request(url, payload, start)
            return await self._non_stream_request(url, payload, start)
        except asyncio.TimeoutError:
            return BenchResult(False, 0, (time.perf_counter() - start) * 1000,
                               error="timeout")
        except Exception as e:
            return BenchResult(False, 0, (time.perf_counter() - start) * 1000,
                               error=str(e))

    async def _non_stream_request(self, url, payload, start) -> BenchResult:
        async with self.session.post(url, json=payload) as resp:
            text = await resp.text()
            latency_ms = (time.perf_counter() - start) * 1000
            if resp.status // 100 != 2:
                return BenchResult(False, resp.status, latency_ms,
                                   error=text[:2000])
            data = json.loads(text)
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices:
                return BenchResult(
                    False, resp.status, latency_ms,
                    error="response contained no completion choices",
                )
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            pt = int(pt) if pt is not None else None
            ct = int(ct) if ct is not None else None
            tt = int(tt) if tt is not None else None
            per_req_tps = (
                ct / (latency_ms / 1000)
                if ct is not None and latency_ms > 0 else None
            )
            return BenchResult(True, resp.status, latency_ms,
                               tokens_prompt=pt, tokens_completion=ct,
                               tokens_total=tt,
                               per_request_tok_per_sec=per_req_tps)

    async def _stream_request(self, url, payload, start) -> BenchResult:
        if self.stream_usage:
            payload = {**payload, "stream_options": {"include_usage": True}}
        ttft_ms = 0.0
        pt = ct = tt = None
        response_status = 0
        saw_choice = False
        saw_finish_reason = False
        saw_done = False

        try:
            async with self.session.post(url, json=payload) as resp:
                response_status = resp.status
                if resp.status // 100 != 2:
                    text = await resp.text()
                    latency_ms = (time.perf_counter() - start) * 1000
                    return BenchResult(False, resp.status, latency_ms,
                                       error=text[:2000])

                while True:
                    line = await resp.content.readline()
                    if not line:
                        break
                    line = line.decode("utf-8", errors="replace").strip()
                    if line.startswith("data: "):
                        payload_str = line[6:].strip()
                    elif line.startswith("data:"):
                        payload_str = line[5:].strip()
                    else:
                        continue

                    if payload_str == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue

                    # TTFT: first content-bearing delta.
                    # Different providers expose content under different keys:
                    #   OpenAI/Anthropic-compat: choices[0].delta.content
                    #   some gateways:           choices[0].delta.text
                    #   reasoning models:
                    #     choices[0].delta.reasoning_content
                    choices = chunk.get("choices", []) or []
                    if (isinstance(choices, list) and choices
                            and isinstance(choices[0], dict)):
                        saw_choice = True
                        choice = choices[0]
                        if choice.get("finish_reason") is not None:
                            saw_finish_reason = True
                        if ttft_ms == 0:
                            delta = choice.get("delta") or {}
                            if isinstance(delta, dict):
                                generated = (
                                    delta.get("content")
                                    or delta.get("text")
                                    or delta.get("reasoning_content")
                                )
                                if generated:
                                    ttft_ms = (
                                        time.perf_counter() - start
                                    ) * 1000

                    # Usage (usually in the last chunk)
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        # Some compatible servers send partial usage updates.
                        # Keep known counts if a later update omits a field.
                        if usage.get("prompt_tokens") is not None:
                            pt = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            ct = int(usage["completion_tokens"])
                        if usage.get("total_tokens") is not None:
                            tt = int(usage["total_tokens"])

                latency_ms = (time.perf_counter() - start) * 1000
                if not saw_choice:
                    return BenchResult(
                        False, resp.status, latency_ms,
                        error="stream contained no completion chunks",
                    )
                if not saw_done and not saw_finish_reason:
                    return BenchResult(
                        False, resp.status, latency_ms,
                        ttft_ms=ttft_ms,
                        tokens_prompt=pt,
                        tokens_completion=ct,
                        tokens_total=tt,
                        error="stream ended before completion",
                    )
                per_req_tps = (
                    ct / (latency_ms / 1000)
                    if latency_ms > 0 and ct is not None else None
                )
                return BenchResult(True, resp.status, latency_ms,
                                   ttft_ms=ttft_ms,
                                   tokens_prompt=pt, tokens_completion=ct,
                                   tokens_total=tt,
                                   per_request_tok_per_sec=per_req_tps)
        except asyncio.TimeoutError:
            return BenchResult(False, response_status,
                               (time.perf_counter() - start) * 1000,
                               ttft_ms=ttft_ms,
                               tokens_prompt=pt,
                               tokens_completion=ct,
                               tokens_total=tt,
                               error="timeout (mid-stream)")
        except Exception as e:
            return BenchResult(False, response_status,
                               (time.perf_counter() - start) * 1000,
                               ttft_ms=ttft_ms,
                               tokens_prompt=pt,
                               tokens_completion=ct,
                               tokens_total=tt,
                               error=f"stream_error: {e}")


async def warmup(client: LLMClient, n: int, max_tokens: int, prompt: str):
    """Send the measurement request shape; exclude these results from metrics."""
    print(f"Warmup ({n} requests)...")
    results = []
    for _ in range(n):
        results.append(await client.chat(prompt, max_tokens=max_tokens))
        await asyncio.sleep(0.1)
    successful = sum(r.success for r in results)
    if successful < n:
        print(f"  Warmup: {successful}/{n} successful; prewarming is not assured.")
    return results


async def run_burst(
    client: LLMClient,
    concurrency: int,
    rounds: int,
    max_tokens: int,
    prompt: str,
) -> tuple:
    """Burst mode: fire concurrency requests at once, repeat rounds times.

    The same fixed prompt is sent to every request so that:
      - per-request throughput comparisons are apples-to-apples;
      - providers that cache prompts may reuse eligible cached prefixes.
    Cache hits are not measured or guaranteed, even after warmup. Later levels
    reuse the same prompt too, so round number does not identify cold/hot data.
    Returns (results: List[BenchResult], elapsed_s: float, peak_req_per_sec: float).

    - elapsed_s: total wall time across all rounds (avg throughput denominator).
    - peak_req_per_sec: throughput of the fastest single round (peak burst).
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    all_results: List[BenchResult] = []
    total_start = time.perf_counter()
    # per-batch (ok_count, elapsed_s) so we can report a real peak throughput
    # instead of conflating it with average across rounds.
    batches: List[tuple] = []

    for _ in range(rounds):
        tasks = [
            asyncio.create_task(
                client.chat(prompt, max_tokens=max_tokens)
            )
            for _ in range(concurrency)
        ]
        batch_start = time.perf_counter()
        batch_results = await asyncio.gather(*tasks)
        batch_elapsed = time.perf_counter() - batch_start
        batch_ok = sum(1 for r in batch_results if r.success)
        batches.append((batch_ok, batch_elapsed))
        all_results.extend(batch_results)

    elapsed_s = time.perf_counter() - total_start
    peak = max((ok / t for ok, t in batches if t > 0), default=0.0)
    return all_results, elapsed_s, peak


async def run_steady(
    client: LLMClient,
    concurrency: int,
    duration_s: int,
    max_tokens: int,
    prompt: str,
) -> tuple:
    """Steady mode: start requests for duration_s seconds, then drain in-flight.

    Returns (results: List[BenchResult], elapsed_s: float), including drain time.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    results: List[BenchResult] = []
    start = time.perf_counter()
    stop_at = start + duration_s

    async def worker():
        while time.perf_counter() < stop_at:
            r = await client.chat(prompt, max_tokens=max_tokens)
            results.append(r)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*tasks)
    elapsed_s = time.perf_counter() - start
    return results, elapsed_s


# ═══════════════════════════════════════════════════════════════════
# 4. Metrics — Statistics
# ═══════════════════════════════════════════════════════════════════

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = min(math.ceil(k), len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize_level(
    results: List[BenchResult],
    concurrency: int,
    elapsed_s: float,
    mode: str = "burst",
    peak_req_per_sec: float = 0.0,
) -> Dict[str, Any]:
    """Compute all metrics for a single concurrency level.

    Throughput semantics:
      - burst:  `req_per_sec` is average across all rounds (total ok / total
                wall time). `peak_req_per_sec` is the fastest single round.
      - steady: `req_per_sec` is completed requests / actual elapsed, including
                the drain of requests started before the dispatch deadline.
                `peak_req_per_sec` is left at 0 (n/a).
    """
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    total = len(results)

    summary: Dict[str, Any] = {
        "concurrency": concurrency,
        "elapsed_s": elapsed_s,
        "total": total,
        "successful": len(ok),
        "failed": len(fail),
        "success_rate": len(ok) / total * 100 if total else 0,
        "error_rate": len(fail) / total * 100 if total else 0,
    }

    if not ok:
        freq: Dict[str, int] = {}
        for r in fail:
            key = f"{r.status_code}: {(r.error or '')[:120]}"
            freq[key] = freq.get(key, 0) + 1
        top = sorted(freq.items(), key=lambda x: -x[1])[:3]
        summary["top_errors"] = [{"key": k, "count": c} for k, c in top]
        return summary

    lats = [r.latency_ms for r in ok]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
    per_req_tps = [r.per_request_tok_per_sec for r in ok]

    summary["avg_latency_ms"] = mean(lats)
    summary["p50_latency_ms"] = percentile(lats, 0.50)
    summary["p95_latency_ms"] = percentile(lats, 0.95)

    if ttfts:
        summary["ttft_avg_ms"] = mean(ttfts)

    summary["req_per_sec"] = len(ok) / elapsed_s if elapsed_s > 0 else 0
    # Burst-only: peak throughput of the fastest round (see run_burst).
    if mode == "burst" and peak_req_per_sec > 0:
        summary["peak_req_per_sec"] = peak_req_per_sec
    prompt_counts = [r.tokens_prompt for r in ok]
    completion_counts = [r.tokens_completion for r in ok]
    prompt_known = all(v is not None for v in prompt_counts)
    completion_known = all(v is not None for v in completion_counts)
    completion_total = sum(completion_counts) if completion_known else None
    summary["completion_tok_per_sec"] = (
        completion_total / elapsed_s
        if completion_known and elapsed_s > 0 else None
    )
    summary["per_request_tok_per_sec"] = (
        mean(per_req_tps)
        if completion_known and all(v is not None for v in per_req_tps) else None
    )

    summary["tokens"] = {
        "prompt_total": sum(prompt_counts) if prompt_known else None,
        "completion_total": completion_total,
        "avg_completion": mean(completion_counts) if completion_known else None,
    }

    if fail:
        freq2: Dict[str, int] = {}
        for r in fail:
            key = f"{r.status_code}: {(r.error or '')[:120]}"
            freq2[key] = freq2.get(key, 0) + 1
        top2 = sorted(freq2.items(), key=lambda x: -x[1])[:3]
        summary["top_errors"] = [{"key": k, "count": c} for k, c in top2]

    return summary


# ═══════════════════════════════════════════════════════════════════
# 5. Reporter — Markdown Generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(config: dict, summaries: list) -> str:
    """Generate report.md with exactly 3 sections:
    Configuration, Overview Table, Conclusion.
    """
    lines: list[str] = []

    # ── Configuration ──
    lines.append("# Tests4LLMs Benchmark Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Model | `{config['model']}` |")
    lines.append(f"| Provider Preset | {config.get('provider') or 'manual'} |")
    lines.append(f"| API | `{config['base_url']}` |")
    lines.append(f"| Mode | {config['mode']} |")
    lines.append(f"| Date | {config['timestamp']} |")
    lines.append(f"| Concurrency Levels | {config['concurrency']} |")
    lines.append(f"| Connection Limit | {config['connection_limit']} |")
    if config["mode"] == "burst":
        lines.append(f"| Rounds | {config['rounds']} per level |")
    else:
        lines.append(f"| Dispatch Window | {config['duration']}s per level |")
    lines.append(f"| Max Tokens | {config['max_tokens']} |")
    lines.append(f"| Token Parameter | `{config['token_param']}` |")
    lines.append(f"| Stream | {'Yes' if config['stream'] else 'No'} |")
    if config["stream"]:
        lines.append(
            f"| Request Stream Usage | {'Yes' if config.get('stream_usage', True) else 'No'} |"
        )
    # JSON notation preserves newlines/whitespace; HTML entities keep arbitrary
    # prompt text from creating Markdown table cells or markup.
    prompt = html.escape(json.dumps(config.get("prompt", ""), ensure_ascii=False))
    prompt = prompt.translate(str.maketrans({
        "|": "&#124;", "`": "&#96;", "\\": "&#92;", "*": "&#42;",
        "_": "&#95;", "[": "&#91;", "]": "&#93;",
    }))
    lines.append(f"| Prompt | {prompt} (JSON string) |")
    lines.append(
        f"| Warmup | {config.get('warmup', 0)} requests once before all levels "
        f"({config.get('warmup_successful', 0)} successful; excluded from metrics) |"
    )
    lines.append(f"| Timeout | {config.get('timeout', 'n/a')}s per request |")
    lines.append(f"| Python Version | {config.get('python_version', 'n/a')} |")
    lines.append(f"| aiohttp Version | {config.get('aiohttp_version', 'n/a')} |")
    lines.append("")
    if config["mode"] == "steady":
        lines.append(
            "The dispatch window limits when new requests start. In-flight "
            "requests finish afterward; elapsed time and req/s include this drain."
        )
        lines.append("")

    # ── Overview Table ──
    has_ttft = any("ttft_avg_ms" in s for s in summaries)
    has_peak = config["mode"] == "burst" and any("peak_req_per_sec" in s for s in summaries)
    lines.append("## Overview")
    lines.append("")

    cols = ["C", "Success%", "Elapsed (s)", "Avg Lat (ms)", "P50 (ms)", "P95 (ms)"]
    if has_ttft:
        cols.append("TTFT (ms)")
    cols.append("req/s (avg)")
    if has_peak:
        cols.append("req/s (peak)")
    cols += ["comp tok/s", "per-req tok/s"]

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")

    for s in summaries:
        c = s["concurrency"]
        sr = f"{s['success_rate']:.0f}%"

        if "avg_latency_ms" in s:
            avg_l = f"{s['avg_latency_ms']:.0f}"
            p50 = f"{s['p50_latency_ms']:.0f}"
            p95 = f"{s['p95_latency_ms']:.0f}"
        else:
            avg_l = p50 = p95 = "-"

        elapsed = f"{s['elapsed_s']:.3f}" if "elapsed_s" in s else "-"
        cells = [str(c), sr, elapsed, avg_l, p50, p95]
        if has_ttft:
            cells.append(f"{s['ttft_avg_ms']:.0f}" if "ttft_avg_ms" in s else "-")
        rps = f"{s['req_per_sec']:.1f}" if s.get("req_per_sec") else "-"
        cells.append(rps)
        if has_peak:
            cells.append(f"{s['peak_req_per_sec']:.1f}"
                         if "peak_req_per_sec" in s else "-")
        ctps = (f"{s['completion_tok_per_sec']:.1f}"
                if s.get("completion_tok_per_sec") is not None else "n/a")
        prtps = (f"{s['per_request_tok_per_sec']:.1f}"
                 if s.get("per_request_tok_per_sec") is not None else "n/a")
        cells += [ctps, prtps]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "Token rates are n/a when completion usage is missing from "
        "any successful request; reported zero tokens remain 0.0."
    )
    lines.append("")

    # ── Conclusion ──
    lines.append("## Conclusion")
    lines.append("")
    lines.append(generate_conclusion(summaries))
    lines.append("")

    return "\n".join(lines)


def generate_conclusion(summaries: list) -> str:
    ok = [s for s in summaries if s["success_rate"] > 0]
    if not ok:
        return "All requests failed. Check API key, base URL, and model name."

    lines: list[str] = []

    # Best low-latency level (success_rate >= 90%)
    with_lat = [s for s in ok
                if "avg_latency_ms" in s and s["success_rate"] >= 90]
    if with_lat:
        best = min(with_lat, key=lambda s: s["avg_latency_ms"])
        lines.append(
            f"- Best low-latency: concurrency={best['concurrency']} "
            f"(avg {best['avg_latency_ms']:.0f}ms)"
        )

    # Best throughput level. In burst mode prefer peak_req_per_sec (the fastest
    # single round); in steady mode fall back to the sustained req/s.
    with_tps = [s for s in ok if s.get("req_per_sec", 0) > 0]
    if with_tps:
        def _throughput_key(s):
            return s.get("peak_req_per_sec") or s.get("req_per_sec", 0)
        best = max(with_tps, key=_throughput_key)
        tps_label = "peak" if "peak_req_per_sec" in best else "avg"
        tps_val = _throughput_key(best)
        comp_tok = (f", {best['completion_tok_per_sec']:.1f} comp tok/s"
                    if best.get("completion_tok_per_sec") else "")
        lines.append(
            f"- Best throughput: concurrency={best['concurrency']} "
            f"({tps_val:.1f} req/s {tps_label}{comp_tok})"
        )

    # Scaling ratio
    if len(with_tps) >= 2:
        first, last = with_tps[0], with_tps[-1]
        if first["req_per_sec"] > 0:
            ratio = last["req_per_sec"] / first["req_per_sec"]
            lines.append(
                f"- Throughput scales {ratio:.1f}x "
                f"from c={first['concurrency']} to c={last['concurrency']}"
            )

    # Success rate inflection point
    all_sorted = sorted(summaries, key=lambda s: s["concurrency"])
    inflection = None
    for s in all_sorted:
        if s["success_rate"] < 95 and s["total"] > 0:
            inflection = s
            break
    if inflection:
        lines.append(
            f"- Success rate drops at: concurrency={inflection['concurrency']} "
            f"({inflection['success_rate']:.0f}%)"
        )

    # TTFT trend
    with_ttft = [s for s in ok if "ttft_avg_ms" in s]
    if len(with_ttft) >= 2:
        first_t, last_t = with_ttft[0], with_ttft[-1]
        if first_t["ttft_avg_ms"] > 0:
            ratio = last_t["ttft_avg_ms"] / first_t["ttft_avg_ms"]
            lines.append(
                f"- TTFT increases {ratio:.1f}x "
                f"from c={first_t['concurrency']} to c={last_t['concurrency']}"
            )

    if not lines:
        lines.append("Benchmark completed. See details above.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 6. IO — File Output
# ═══════════════════════════════════════════════════════════════════

def save_json(data: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_report(md: str, output_dir: str) -> str:
    path = os.path.join(output_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

async def run(args):
    apply_provider(args)
    levels = parse_concurrency(args.concurrency)
    validate_args(args)
    api_key = resolve_api_key(args)
    connection_limit = max(levels)

    config = {
        "base_url": args.base_url,
        "provider": args.provider.lower() if args.provider else "manual",
        "model": args.model,
        "token_param": args.token_param,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "connection_limit": connection_limit,
        "rounds": args.rounds,
        "duration": args.duration,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "stream_usage": args.stream_usage,
        "prompt": args.prompt,
        "warmup": args.warmup,
        "timeout": args.timeout,
        "python_version": platform.python_version(),
        "aiohttp_version": aiohttp.__version__,
        "timestamp": datetime.now().astimezone().isoformat(),
    }

    print("=" * 55)
    print("Tests4LLMs Benchmark")
    print("=" * 55)
    print(f"  API:      {config['base_url']}")
    print(f"  Model:    {config['model']}")
    print(f"  Token:    {config['token_param']}")
    print(f"  Mode:     {config['mode']}")
    print(f"  Levels:   {levels}")
    print(f"  Pool:     {config['connection_limit']} connections")
    if config["mode"] == "burst":
        print(f"  Rounds:   {config['rounds']} per level")
    else:
        print(f"  Dispatch: {config['duration']}s per level, then drain")
    print(f"  Stream:   {config['stream']}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    all_raw: Dict[str, list] = {}
    summaries: list = []

    async with LLMClient(
        config["base_url"], api_key, config["model"],
        timeout_s=args.timeout, stream=config["stream"],
        max_connections=config["connection_limit"],
        token_param=config["token_param"],
        stream_usage=config["stream_usage"],
    ) as client:

        warmup_results = await warmup(
            client, args.warmup, config["max_tokens"], config["prompt"],
        )
        config["warmup_successful"] = sum(r.success for r in warmup_results)

        for lvl in levels:
            peak = 0.0
            if config["mode"] == "burst":
                results, elapsed, peak = await run_burst(
                    client, lvl, args.rounds,
                    config["max_tokens"], config["prompt"],
                )
            else:
                results, elapsed = await run_steady(
                    client, lvl, args.duration,
                    config["max_tokens"], config["prompt"],
                )

            s = summarize_level(list(results), lvl, elapsed,
                                mode=config["mode"],
                                peak_req_per_sec=peak)
            ok_count = s["successful"]
            total_count = s["total"]

            # Print inline progress
            msg = f"  c={lvl}  OK={ok_count}/{total_count}  elapsed={elapsed:.3f}s"
            if ok_count > 0 and "avg_latency_ms" in s:
                comp_tps = s["completion_tok_per_sec"]
                comp_label = f"{comp_tps:.1f}" if comp_tps is not None else "n/a"
                msg += (f"  avg={s['avg_latency_ms']:.0f}ms"
                        f"  req/s={s['req_per_sec']:.1f}"
                        f"  comp_tok/s={comp_label}")
            print(msg)

            all_raw[f"c{lvl}"] = [asdict(r) for r in results]
            summaries.append(s)

    # ── Save outputs ──
    output = {"config": config, "summaries": summaries, "raw": all_raw}
    json_path = save_json(output, args.output_dir)
    print(f"\nSaved: {json_path}")

    md = generate_report(config, summaries)
    md_path = save_report(md, args.output_dir)
    print(f"Saved: {md_path}")


def main():
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
