#!/usr/bin/env python3
"""
Tests4LLMs — LLM API Benchmark (Single Script)

Outputs: result.json (raw details) + report.md (summary with tables)

Usage:
  pip install aiohttp

  python tests4llms.py \\
    --base-url https://api.openai.com/v1 \\
    --api-key-env OPENAI_API_KEY \\
    --model gpt-4o \\
    --concurrency 1,5,10,20 \\
    --rounds 10 \\
    --max-tokens 100

  # With streaming (TTFT measurement)
  python tests4llms.py \\
    --base-url https://api.openai.com/v1 \\
    --api-key sk-xxx \\
    --model gpt-4o \\
    --concurrency 1,5,10 \\
    --rounds 10 \\
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


# ═══════════════════════════════════════════════════════════════
# 1. 配置解析 — CLI Argument Parsing
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tests4LLMs — LLM API Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tests4llms.py --base-url https://api.openai.com/v1 "
            "--api-key-env OPENAI_API_KEY --model gpt-4o\n"
            "  python tests4llms.py --base-url https://open.bigmodel.cn/api/paas/v4 "
            "--api-key-env ZHIPU_API_KEY --model glm-4-flash --stream\n"
        ),
    )

    g = p.add_argument_group("API")
    g.add_argument("--base-url", required=True,
                   help="API base URL (OpenAI-compatible)")
    g.add_argument("--api-key", default="",
                   help="API key (or use --api-key-env)")
    g.add_argument("--api-key-env", default="",
                   help="Env var name holding the API key")
    g.add_argument("--model", required=True,
                   help="Model name to benchmark")

    g = p.add_argument_group("Benchmark")
    g.add_argument("--concurrency", default="1,5,10,20",
                   help="Comma-separated concurrency levels (default: 1,5,10,20)")
    g.add_argument("--rounds", type=int, default=10,
                   help="Requests per concurrency level (default: 10)")
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


# ═══════════════════════════════════════════════════════════════
# 2. 请求发送 — Async LLM Client
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    success: bool
    status_code: int
    latency_ms: float
    ttft_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    tps: float = 0.0
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
            tps = ct / (latency_ms / 1000) if latency_ms > 0 else 0
            return BenchResult(True, resp.status, latency_ms,
                               tokens_prompt=pt, tokens_completion=ct,
                               tokens_total=tt, tps=tps)

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
            tps = ct / (latency_ms / 1000) if latency_ms > 0 and ct > 0 else 0
            return BenchResult(True, resp.status, latency_ms,
                               ttft_ms=ttft_ms,
                               tokens_prompt=pt, tokens_completion=ct,
                               tokens_total=tt, tps=tps)


async def warmup(client: LLMClient, n: int, max_tokens: int):
    print(f"Warmup ({n} requests)...")
    for _ in range(n):
        await client.chat("Say 'ok'.", max_tokens=min(max_tokens, 10))
        await asyncio.sleep(0.1)


async def run_level(
    client: LLMClient,
    concurrency: int,
    rounds: int,
    max_tokens: int,
    prompt: str,
) -> List[BenchResult]:
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> BenchResult:
        async with sem:
            return await client.chat(f"{prompt} (#{i+1})", max_tokens=max_tokens)

    tasks = [asyncio.create_task(one(i)) for i in range(rounds)]
    return list(await asyncio.gather(*tasks))


# ═══════════════════════════════════════════════════════════════
# 3. 指标统计 — Metrics & Statistics
# ═══════════════════════════════════════════════════════════════

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


def summarize_level(results: List[BenchResult], concurrency: int) -> Dict[str, Any]:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    total = len(results)

    summary: Dict[str, Any] = {
        "concurrency": concurrency,
        "total": total,
        "successful": len(ok),
        "failed": len(fail),
        "success_rate": len(ok) / total * 100 if total else 0,
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
    tps_list = [r.tps for r in ok if r.tps > 0]

    summary["latency"] = {
        "min_ms": min(lats),
        "avg_ms": mean(lats),
        "p50_ms": percentile(lats, 0.50),
        "p95_ms": percentile(lats, 0.95),
        "max_ms": max(lats),
    }

    if ttfts:
        summary["ttft"] = {
            "avg_ms": mean(ttfts),
            "p50_ms": percentile(ttfts, 0.50),
            "p95_ms": percentile(ttfts, 0.95),
        }

    summary["tps"] = {
        "avg": mean(tps_list) if tps_list else 0,
        "max": max(tps_list) if tps_list else 0,
    }

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


# ═══════════════════════════════════════════════════════════════
# 4. 报告生成 — JSON & Markdown
# ═══════════════════════════════════════════════════════════════

def save_result_json(data: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def generate_markdown(config: dict, summaries: list) -> str:
    lines: list[str] = []

    # ── Header ──
    lines.append("# Tests4LLMs Benchmark Report")
    lines.append("")

    # ── Configuration ──
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Model | `{config['model']}` |")
    lines.append(f"| API | `{config['base_url']}` |")
    lines.append(f"| Date | {config['timestamp']} |")
    lines.append(f"| Rounds | {config['rounds']} per level |")
    lines.append(f"| Max Tokens | {config['max_tokens']} |")
    lines.append(f"| Stream | {'Yes' if config['stream'] else 'No'} |")
    lines.append(f"| Concurrency Levels | {', '.join(str(s['concurrency']) for s in summaries)} |")
    lines.append("")

    # ── Concurrency Comparison Table ──
    has_ttft = any("ttft" in s for s in summaries)

    lines.append("## Concurrency Comparison")
    lines.append("")
    hdr = "| C | Success | Avg Latency (ms) | P50 (ms) | P95 (ms) | Output tok/s |"
    sep = "|---|---------|------------------|----------|----------|-------------|"
    if has_ttft:
        hdr = "| C | Success | Avg Latency (ms) | P50 (ms) | P95 (ms) | TTFT (ms) | Output tok/s |"
        sep = "|---|---------|------------------|----------|----------|-----------|-------------|"
    lines.append(hdr)
    lines.append(sep)

    for s in summaries:
        c = s["concurrency"]
        sr = f"{s['successful']}/{s['total']}"

        if "latency" in s:
            lat = s["latency"]
            avg_l = f"{lat['avg_ms']:.0f}"
            p50 = f"{lat['p50_ms']:.0f}"
            p95 = f"{lat['p95_ms']:.0f}"
        else:
            avg_l = p50 = p95 = "-"

        tps_val = f"{s['tps']['avg']:.1f}" if "tps" in s and s["tps"]["avg"] > 0 else "-"

        if has_ttft:
            ttft_val = f"{s['ttft']['avg_ms']:.0f}" if "ttft" in s else "-"
            row = f"| {c} | {sr} | {avg_l} | {p50} | {p95} | {ttft_val} | {tps_val} |"
        else:
            row = f"| {c} | {sr} | {avg_l} | {p50} | {p95} | {tps_val} |"
        lines.append(row)

    lines.append("")

    # ── Per-Level Details ──
    lines.append("## Per-Level Details")
    lines.append("")

    for s in summaries:
        c = s["concurrency"]
        lines.append(f"### Concurrency = {c}")
        lines.append("")

        if "latency" not in s:
            lines.append(f"All {s['total']} requests failed.")
            for e in s.get("top_errors", []):
                lines.append(f"- `{e['key']}` (x{e['count']})")
            lines.append("")
            continue

        lat = s["latency"]
        lines.append(f"- Success rate: **{s['success_rate']:.1f}%** "
                     f"({s['successful']}/{s['total']})")
        lines.append(f"- Latency: min={lat['min_ms']:.0f}ms, "
                     f"avg={lat['avg_ms']:.0f}ms, "
                     f"p50={lat['p50_ms']:.0f}ms, "
                     f"p95={lat['p95_ms']:.0f}ms, "
                     f"max={lat['max_ms']:.0f}ms")

        if "ttft" in s:
            t = s["ttft"]
            lines.append(f"- TTFT: avg={t['avg_ms']:.0f}ms, "
                         f"p50={t['p50_ms']:.0f}ms, "
                         f"p95={t['p95_ms']:.0f}ms")

        if "tps" in s and s["tps"]["avg"] > 0:
            lines.append(f"- Output throughput: avg={s['tps']['avg']:.1f} tok/s, "
                         f"max={s['tps']['max']:.1f} tok/s")

        if "tokens" in s:
            tok = s["tokens"]
            lines.append(f"- Tokens: {tok['completion_total']} completion tokens total, "
                         f"avg={tok['avg_completion']:.1f}/req")

        if s.get("top_errors"):
            err_str = ", ".join(f"`{e['key']}` (x{e['count']})"
                                for e in s["top_errors"])
            lines.append(f"- Errors: {err_str}")

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

    # Best latency
    with_lat = [s for s in ok if "latency" in s]
    if with_lat:
        best = min(with_lat, key=lambda s: s["latency"]["avg_ms"])
        lines.append(f"- Lowest avg latency at concurrency={best['concurrency']}: "
                     f"{best['latency']['avg_ms']:.0f}ms")

    # Best throughput
    with_tps = [s for s in ok if s.get("tps", {}).get("avg", 0) > 0]
    if with_tps:
        best = max(with_tps, key=lambda s: s["tps"]["avg"])
        lines.append(f"- Highest throughput at concurrency={best['concurrency']}: "
                     f"{best['tps']['avg']:.1f} tok/s")

    # Scaling
    if len(with_tps) >= 2:
        first, last = with_tps[0], with_tps[-1]
        f_tps, l_tps = first["tps"]["avg"], last["tps"]["avg"]
        if f_tps > 0:
            lines.append(f"- Throughput scales {l_tps / f_tps:.1f}x "
                         f"from c={first['concurrency']} to c={last['concurrency']}")

    # Success rate warnings
    failing = [s for s in summaries if 0 < s["success_rate"] < 100]
    if failing:
        parts = [f"c={s['concurrency']} ({s['success_rate']:.0f}%)"
                 for s in failing]
        lines.append(f"- Warning: success rate dropped at {', '.join(parts)}")

    # TTFT trend
    with_ttft = [s for s in ok if "ttft" in s]
    if len(with_ttft) >= 2:
        first_t, last_t = with_ttft[0], with_ttft[-1]
        f_t = first_t["ttft"]["avg_ms"]
        l_t = last_t["ttft"]["avg_ms"]
        if f_t > 0:
            lines.append(f"- TTFT increases {l_t / f_t:.1f}x "
                         f"from c={first_t['concurrency']} to c={last_t['concurrency']}")

    if not lines:
        lines.append("Benchmark completed. See details above.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def run(args):
    api_key = resolve_api_key(args)
    levels = parse_concurrency(args.concurrency)

    config = {
        "base_url": args.base_url,
        "model": args.model,
        "rounds": args.rounds,
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
    print(f"  Levels:   {levels}")
    print(f"  Rounds:   {config['rounds']}  Stream: {config['stream']}")
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
            print(f"  c={lvl} ({config['rounds']} rounds) ... ", end="", flush=True)
            results = await run_level(
                client, lvl, config["rounds"],
                config["max_tokens"], config["prompt"],
            )
            ok_count = sum(1 for r in results if r.success)
            msg = f"OK={ok_count}/{config['rounds']}"
            if ok_count > 0:
                ok_r = [r for r in results if r.success]
                avg_lat = mean([r.latency_ms for r in ok_r])
                avg_tps = mean([r.tps for r in ok_r if r.tps > 0]) \
                    if any(r.tps > 0 for r in ok_r) else 0
                msg += f"  avg={avg_lat:.0f}ms  tok/s={avg_tps:.1f}"
            print(msg)

            all_raw[f"c{lvl}"] = [asdict(r) for r in results]
            summaries.append(summarize_level(list(results), lvl))

    # ── Save result.json ──
    output = {"config": config, "summaries": summaries, "raw": all_raw}
    json_path = save_result_json(output, args.output_dir)
    print(f"\nSaved: {json_path}")

    # ── Generate report.md ──
    md = generate_markdown(config, summaries)
    md_path = os.path.join(args.output_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved: {md_path}")


def main():
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
