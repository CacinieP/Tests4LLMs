#!/usr/bin/env python3
"""
Tests4LLMs — LLM API Benchmark (Single Script)

Outputs: result.json (raw details) + report.md (summary with tables)

支持的 API（OpenAI 兼容格式），使用 --provider 快捷切换或手动指定参数：

## 1. 智谱 Zhipu AI
# python tests4llms.py --provider zhipu
# 或手动指定:
# python tests4llms.py --base-url https://open.bigmodel.cn/api/paas/v4 --api-key-env ZHIPU_API_KEY --model glm-5

## 2. MiniMax
# python tests4llms.py --provider minimax

## 3. Kimi（月之暗面 Moonshot）
# python tests4llms.py --provider kimi

## 4. StepFun（阶跃星辰）
# python tests4llms.py --provider stepfun

## 5. Google Gemini
# python tests4llms.py --provider gemini

## 6. OpenAI
# python tests4llms.py --provider openai

## 7. Anthropic
# python tests4llms.py --provider anthropic

Usage:
  pip install aiohttp

  # Burst mode: 瞬时并发测试
  python tests4llms.py \\
    --provider openai \\
    --concurrency 1,5,10,20 \\
    --rounds 3

  # Steady mode: 持续吞吐测试
  python tests4llms.py \\
    --provider openai \\
    --mode steady \\
    --concurrency 5,10 \\
    --duration 30

  # Streaming (TTFT measurement)
  python tests4llms.py \\
    --provider openai \\
    --concurrency 1,5,10 \\
    --stream
"""

import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, List, Optional

import aiohttp


# ═══════════════════════════════════════════════════════════════════
# 1. CLI — Argument Parsing
# ═══════════════════════════════════════════════════════════════════

PROVIDER_NAMES = "zhipu / minimax / kimi / stepfun / gemini / openai / anthropic"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tests4LLMs — LLM API Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tests4llms.py --provider zhipu --concurrency 1,5,10\n"
            "  python tests4llms.py --provider openai --mode steady "
            "--concurrency 5,10 --duration 30\n"
            "  python tests4llms.py --base-url https://api.openai.com/v1 "
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
                   help=f"Provider preset: {PROVIDER_NAMES}")

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
    g.add_argument("--prompt", default="Count from 1 to 10.",
                   help="Test prompt (default: 'Count from 1 to 10.')")

    g = p.add_argument_group("Tuning")
    g.add_argument("--timeout", type=float, default=120.0,
                   help="Request timeout in seconds (default: 120)")
    g.add_argument("--warmup", type=int, default=2,
                   help="Warmup requests before benchmark (default: 2)")
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
    return sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))


# ═══════════════════════════════════════════════════════════════════
# 2. Provider Profile — Endpoint Presets
# ═══════════════════════════════════════════════════════════════════

PROVIDERS: Dict[str, Dict[str, str]] = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "ZHIPU_API_KEY",
        "model": "glm-5",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "key_env": "MINIMAX_API_KEY",
        "model": "MiniMax-M2.7",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "key_env": "MOONSHOT_API_KEY",
        "model": "kimi-k2.5",
    },
    "stepfun": {
        "base_url": "https://api.stepfun.com/v1",
        "key_env": "STEP_API_KEY",
        "model": "step-3.5-flash",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.1-pro-preview",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "key_env": "ANTHROPIC_API_KEY",
        "model": "claude-opus-4-6",
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


def validate_args(args):
    """Validate required args after provider resolution."""
    if not args.base_url:
        raise SystemExit("ERROR: --base-url is required (or use --provider)")
    if not args.model:
        raise SystemExit("ERROR: --model is required (or use --provider)")


# ═══════════════════════════════════════════════════════════════════
# 3. Runner — Test Execution
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    success: bool
    status_code: int
    latency_ms: float
    ttft_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    per_request_tok_per_sec: float = 0.0
    error: Optional[str] = None


class LLMClient:
    """Async LLM client with connection pooling (OpenAI-compatible)."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout_s: float = 120.0, stream: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.connector = aiohttp.TCPConnector(
            limit=256, limit_per_host=256,
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
            "max_tokens": max_tokens,
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
            latency_ms = (time.perf_counter() - start) * 1000
            if resp.status // 100 != 2:
                text = await resp.text()
                return BenchResult(False, resp.status, latency_ms,
                                   error=text[:2000])
            data = json.loads(await resp.text())
            usage = data.get("usage") or {}
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            tt = int(usage.get("total_tokens", 0) or 0)
            per_req_tps = ct / (latency_ms / 1000) if latency_ms > 0 else 0
            return BenchResult(True, resp.status, latency_ms,
                               tokens_prompt=pt, tokens_completion=ct,
                               tokens_total=tt,
                               per_request_tok_per_sec=per_req_tps)

    async def _stream_request(self, url, payload, start) -> BenchResult:
        payload = {**payload, "stream_options": {"include_usage": True}}
        async with self.session.post(url, json=payload) as resp:
            if resp.status // 100 != 2:
                latency_ms = (time.perf_counter() - start) * 1000
                text = await resp.text()
                return BenchResult(False, resp.status, latency_ms,
                                   error=text[:2000])

            ttft_ms = 0.0
            pt = ct = tt = 0

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
                    continue
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                # TTFT: first content-bearing delta
                if ttft_ms == 0:
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content")
                        if content:
                            ttft_ms = (time.perf_counter() - start) * 1000

                # Usage (usually in the last chunk)
                usage = chunk.get("usage")
                if usage:
                    pt = int(usage.get("prompt_tokens", 0) or 0)
                    ct = int(usage.get("completion_tokens", 0) or 0)
                    tt = int(usage.get("total_tokens", 0) or 0)

            latency_ms = (time.perf_counter() - start) * 1000
            per_req_tps = ct / (latency_ms / 1000) if latency_ms > 0 and ct > 0 else 0
            return BenchResult(True, resp.status, latency_ms,
                               ttft_ms=ttft_ms,
                               tokens_prompt=pt, tokens_completion=ct,
                               tokens_total=tt,
                               per_request_tok_per_sec=per_req_tps)


async def warmup(client: LLMClient, n: int, max_tokens: int):
    print(f"Warmup ({n} requests)...")
    for _ in range(n):
        await client.chat("Say 'ok'.", max_tokens=min(max_tokens, 10))
        await asyncio.sleep(0.1)


async def run_burst(
    client: LLMClient,
    concurrency: int,
    rounds: int,
    max_tokens: int,
    prompt: str,
) -> tuple:
    """Burst mode: fire concurrency requests at once, repeat rounds times.

    Returns (results: List[BenchResult], elapsed_s: float).
    """
    all_results: List[BenchResult] = []
    total_start = time.perf_counter()

    for batch in range(rounds):
        tasks = [
            asyncio.create_task(
                client.chat(f"{prompt} (#{i+1})", max_tokens=max_tokens)
            )
            for i in range(concurrency)
        ]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

    elapsed_s = time.perf_counter() - total_start
    return all_results, elapsed_s


async def run_steady(
    client: LLMClient,
    concurrency: int,
    duration_s: int,
    max_tokens: int,
    prompt: str,
) -> tuple:
    """Steady mode: concurrency workers sustain load for duration_s seconds.

    Returns (results: List[BenchResult], elapsed_s: float).
    """
    results: List[BenchResult] = []
    stop_at = time.perf_counter() + duration_s
    counter = 0
    lock = asyncio.Lock()

    async def worker(wid: int):
        nonlocal counter
        while time.perf_counter() < stop_at:
            async with lock:
                counter += 1
                req_id = counter
            r = await client.chat(
                f"{prompt} (#{req_id})", max_tokens=max_tokens
            )
            results.append(r)
            # small pacing: avoid ultra-bursting
            await asyncio.sleep(0.05 if r.success else 0.2)

    start = time.perf_counter()
    tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
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
) -> Dict[str, Any]:
    """Compute all metrics for a single concurrency level."""
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    total = len(results)

    summary: Dict[str, Any] = {
        "concurrency": concurrency,
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
    per_req_tps = [r.per_request_tok_per_sec for r in ok
                   if r.per_request_tok_per_sec > 0]

    summary["avg_latency_ms"] = mean(lats)
    summary["p50_latency_ms"] = percentile(lats, 0.50)
    summary["p95_latency_ms"] = percentile(lats, 0.95)

    if ttfts:
        summary["ttft_avg_ms"] = mean(ttfts)

    summary["req_per_sec"] = len(ok) / elapsed_s if elapsed_s > 0 else 0
    summary["completion_tok_per_sec"] = (
        sum(r.tokens_completion for r in ok) / elapsed_s
        if elapsed_s > 0 else 0
    )
    summary["per_request_tok_per_sec"] = (
        mean(per_req_tps) if per_req_tps else 0
    )

    summary["tokens"] = {
        "prompt_total": sum(r.tokens_prompt for r in ok),
        "completion_total": sum(r.tokens_completion for r in ok),
        "avg_completion": mean([r.tokens_completion for r in ok]),
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
    lines.append(f"| API | `{config['base_url']}` |")
    lines.append(f"| Mode | {config['mode']} |")
    lines.append(f"| Date | {config['timestamp']} |")
    lines.append(f"| Concurrency Levels | {config['concurrency']} |")
    if config["mode"] == "burst":
        lines.append(f"| Rounds | {config['rounds']} per level |")
    else:
        lines.append(f"| Duration | {config['duration']}s |")
    lines.append(f"| Max Tokens | {config['max_tokens']} |")
    lines.append(f"| Stream | {'Yes' if config['stream'] else 'No'} |")
    lines.append("")

    # ── Overview Table ──
    has_ttft = any("ttft_avg_ms" in s for s in summaries)
    lines.append("## Overview")
    lines.append("")

    if has_ttft:
        lines.append("| C | Success% | Avg Lat (ms) | P50 (ms) | P95 (ms) |"
                     " TTFT (ms) | req/s | comp tok/s | per-req tok/s |")
        lines.append("|---|----------|-------------|----------|----------|"
                     "-----------|-------|------------|---------------|")
    else:
        lines.append("| C | Success% | Avg Lat (ms) | P50 (ms) | P95 (ms) |"
                     " req/s | comp tok/s | per-req tok/s |")
        lines.append("|---|----------|-------------|----------|----------|"
                     "-------|------------|---------------|")

    for s in summaries:
        c = s["concurrency"]
        sr = f"{s['success_rate']:.0f}%"

        if "avg_latency_ms" in s:
            avg_l = f"{s['avg_latency_ms']:.0f}"
            p50 = f"{s['p50_latency_ms']:.0f}"
            p95 = f"{s['p95_latency_ms']:.0f}"
        else:
            avg_l = p50 = p95 = "-"

        rps = f"{s['req_per_sec']:.1f}" if s.get("req_per_sec") else "-"
        ctps = (f"{s['completion_tok_per_sec']:.1f}"
                if s.get("completion_tok_per_sec") else "-")
        prtps = (f"{s['per_request_tok_per_sec']:.1f}"
                 if s.get("per_request_tok_per_sec") else "-")

        if has_ttft:
            ttft = f"{s['ttft_avg_ms']:.0f}" if "ttft_avg_ms" in s else "-"
            row = (f"| {c} | {sr} | {avg_l} | {p50} | {p95} |"
                   f" {ttft} | {rps} | {ctps} | {prtps} |")
        else:
            row = (f"| {c} | {sr} | {avg_l} | {p50} | {p95} |"
                   f" {rps} | {ctps} | {prtps} |")
        lines.append(row)

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

    # Best throughput level
    with_tps = [s for s in ok if s.get("req_per_sec", 0) > 0]
    if with_tps:
        best = max(with_tps, key=lambda s: s["req_per_sec"])
        comp_tok = (f", {best['completion_tok_per_sec']:.1f} comp tok/s"
                    if best.get("completion_tok_per_sec") else "")
        lines.append(
            f"- Best throughput: concurrency={best['concurrency']} "
            f"({best['req_per_sec']:.1f} req/s{comp_tok})"
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
    validate_args(args)
    api_key = resolve_api_key(args)
    levels = parse_concurrency(args.concurrency)

    config = {
        "base_url": args.base_url,
        "model": args.model,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "duration": args.duration,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "prompt": args.prompt,
        "timestamp": datetime.now().isoformat(),
    }

    print("=" * 55)
    print("Tests4LLMs Benchmark")
    print("=" * 55)
    print(f"  API:      {config['base_url']}")
    print(f"  Model:    {config['model']}")
    print(f"  Mode:     {config['mode']}")
    print(f"  Levels:   {levels}")
    if config["mode"] == "burst":
        print(f"  Rounds:   {config['rounds']} per level")
    else:
        print(f"  Duration: {config['duration']}s per level")
    print(f"  Stream:   {config['stream']}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    all_raw: Dict[str, list] = {}
    summaries: list = []

    async with LLMClient(
        config["base_url"], api_key, config["model"],
        timeout_s=args.timeout, stream=config["stream"],
    ) as client:

        await warmup(client, args.warmup, config["max_tokens"])

        for lvl in levels:
            if config["mode"] == "burst":
                results, elapsed = await run_burst(
                    client, lvl, args.rounds,
                    config["max_tokens"], config["prompt"],
                )
            else:
                results, elapsed = await run_steady(
                    client, lvl, args.duration,
                    config["max_tokens"], config["prompt"],
                )

            s = summarize_level(list(results), lvl, elapsed)
            ok_count = s["successful"]
            total_count = s["total"]

            # Print inline progress
            msg = f"  c={lvl}  OK={ok_count}/{total_count}"
            if ok_count > 0 and "avg_latency_ms" in s:
                msg += (f"  avg={s['avg_latency_ms']:.0f}ms"
                        f"  req/s={s['req_per_sec']:.1f}"
                        f"  comp_tok/s={s['completion_tok_per_sec']:.1f}")
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
