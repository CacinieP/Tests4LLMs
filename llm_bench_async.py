#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Bench (Async, connection pooling)

支持的 7 家 API（OpenAI 兼容格式），取消对应注释并修改下方 BASE_URL / API_KEY_ENV / DEFAULT_MODEL 即可切换：

## 1. 智谱 Zhipu AI
# export ZHIPU_API_KEY="..."
# BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
# DEFAULT_MODEL = "glm-5"

## 2. MiniMax
# export MINIMAX_API_KEY="..."
# BASE_URL = "https://api.minimax.io/v1"        # 中国区常用: https://api.minimaxi.com/v1
# DEFAULT_MODEL = "MiniMax-M2.7"

## 3. Kimi（月之暗面 Moonshot）
# export MOONSHOT_API_KEY="..."
# BASE_URL = "https://api.moonshot.cn/v1"
# DEFAULT_MODEL = "kimi-k2.5"

## 4. StepFun（阶跃星辰）
# export STEP_API_KEY="..."
# BASE_URL = "https://api.stepfun.com/v1"
# DEFAULT_MODEL = "step-3.5-flash"

## 5. Google Gemini
# export GEMINI_API_KEY="..."
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# DEFAULT_MODEL = "gemini-3.1-pro-preview"     # 兼容页示例常用: gemini-3-flash-preview

## 6. OpenAI
# export OPENAI_API_KEY="..."
# BASE_URL = "https://api.openai.com/v1"
# DEFAULT_MODEL = "gpt-5.4"

## 7. Anthropic
# export ANTHROPIC_API_KEY="..."
# BASE_URL = "https://api.anthropic.com/v1/"
# DEFAULT_MODEL = "claude-opus-4-6"

Features:
- asyncio + aiohttp connection pooling (keep-alive)
- latency stats: min/avg/median/p95/max
- TPS (completion tokens / latency), RPM/TPM over a duration window
- concurrency sweep
- warmup
- saves JSON (+ optional CSV)

Usage:
  pip install aiohttp
  # 取消上方对应 provider 注释，修改 BASE_URL / API_KEY_ENV / DEFAULT_MODEL
  export ZHIPU_API_KEY="..."
  python llm_bench_async.py

Optional args:
  python llm_bench_async.py --model glm-5 --levels 1,2,5,10,20,30,40,50 --duration 30 --outdir test-results
"""

import os
import json
import time
import math
import argparse
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# ============================================================
# 切换 API 时只需修改以下三行 + 上方注释区取消对应注释
# ============================================================
# 默认使用智谱 Zhipu AI
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
API_KEY_ENV = "ZHIPU_API_KEY"
DEFAULT_MODEL = "glm-5"


@dataclass
class TestResult:
    success: bool
    status_code: int
    latency_ms: float
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    tps: float = 0.0
    error: Optional[str] = None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    # linear interpolation
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def summarize(results: List[TestResult], name: str) -> Dict[str, Any]:
    total = len(results)
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    if not ok:
        return {
            "test": name,
            "total_requests": total,
            "successful": 0,
            "failed": len(fail),
            "success_rate": 0.0,
            "error": "No successful requests",
        }

    lat = [r.latency_ms for r in ok]
    tok_total = [r.tokens_total for r in ok]
    tps_list = [r.tps for r in ok if r.tps > 0]

    return {
        "test": name,
        "total_requests": total,
        "successful": len(ok),
        "failed": len(fail),
        "success_rate": (len(ok) / total * 100.0) if total else 0.0,
        "latency": {
            "min_ms": min(lat),
            "avg_ms": mean(lat),
            "median_ms": median(lat),
            "p95_ms": percentile(lat, 0.95),
            "max_ms": max(lat),
        },
        "tokens": {
            "total": sum(tok_total),
            "avg_per_request": mean(tok_total),
        },
        "tps": {
            "avg": mean(tps_list) if tps_list else 0.0,
            "max": max(tps_list) if tps_list else 0.0,
        },
        "errors_top": top_errors(fail, k=3),
    }


def top_errors(failed: List[TestResult], k: int = 3) -> List[Dict[str, Any]]:
    if not failed:
        return []
    freq: Dict[str, int] = {}
    for r in failed:
        key = f"{r.status_code}:{(r.error or '')[:120]}"
        freq[key] = freq.get(key, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]
    out = []
    for key, count in items:
        status, msg = key.split(":", 1)
        out.append({"status_code": int(status), "count": count, "sample": msg})
    return out


class LLMClient:
    """OpenAI 兼容格式的异步 LLM 压测客户端"""
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        max_connections: int = 200,
        keepalive_connections: int = 100,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)

        # Connection pooling + keep-alive
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.session = aiohttp.ClientSession(
            timeout=self.timeout, connector=self.connector, headers=headers
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
        self.session = None

    async def chat(self, prompt: str, max_tokens: int = 100) -> TestResult:
        assert self.session is not None, "ClientSession not initialized"

        url = f"{BASE_URL}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            async with self.session.post(url, json=payload) as resp:
                latency_ms = (time.perf_counter() - start) * 1000.0
                status = resp.status
                text = await resp.text()

                if status // 100 != 2:
                    return TestResult(
                        success=False,
                        status_code=status,
                        latency_ms=latency_ms,
                        error=text[:2000],
                    )

                data = json.loads(text)
                usage = data.get("usage", {}) or {}
                tokens_prompt = int(usage.get("prompt_tokens", 0) or 0)
                tokens_completion = int(usage.get("completion_tokens", 0) or 0)
                tokens_total = int(usage.get("total_tokens", 0) or 0)
                tps = (tokens_completion / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

                return TestResult(
                    success=True,
                    status_code=status,
                    latency_ms=latency_ms,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    tokens_total=tokens_total,
                    tps=tps,
                )
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return TestResult(False, 0, latency_ms, error="timeout")
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return TestResult(False, 0, latency_ms, error=str(e))


async def run_latency_test(client: LLMClient, n: int = 10) -> List[TestResult]:
    print(f"\n{'='*60}\nLatency Test (sequential) n={n}\n{'='*60}")
    out: List[TestResult] = []
    for i in range(n):
        r = await client.chat(f"Say 'hello' and nothing else. Request #{i+1}", max_tokens=20)
        out.append(r)
        if r.success:
            print(f"  {i+1:02d}/{n}: OK  {r.latency_ms:7.0f}ms  comp={r.tokens_completion:4d}  tps={r.tps:6.1f}")
        else:
            print(f"  {i+1:02d}/{n}: FAIL status={r.status_code} err={((r.error or '')[:80])}")
        await asyncio.sleep(0.2)
    return out


async def run_concurrency_level(
    client: LLMClient,
    level: int,
    max_tokens: int = 50,
) -> Tuple[List[TestResult], float]:
    sem = asyncio.Semaphore(level)

    async def one(i: int) -> TestResult:
        async with sem:
            return await client.chat(f"Count from 1 to 5. Request #{i+1}", max_tokens=max_tokens)

    start = time.perf_counter()
    tasks = [asyncio.create_task(one(i)) for i in range(level)]
    results = await asyncio.gather(*tasks)
    total_s = time.perf_counter() - start
    return results, total_s


async def run_concurrency_sweep(
    client: LLMClient,
    levels: List[int],
) -> Dict[int, Dict[str, Any]]:
    print(f"\n{'='*60}\nConcurrency Sweep\n{'='*60}")
    all_out: Dict[int, Dict[str, Any]] = {}

    for lvl in levels:
        print(f"\n  Concurrency level: {lvl}")
        results, total_s = await run_concurrency_level(client, lvl, max_tokens=50)
        ok = [r for r in results if r.success]
        print(f"    Total time: {total_s:.2f}s")
        print(f"    Successful: {len(ok)}/{lvl}")

        if ok:
            avg_lat = mean([r.latency_ms for r in ok])
            total_tok = sum([r.tokens_total for r in ok])
            avg_tps = mean([r.tps for r in ok if r.tps > 0]) if any(r.tps > 0 for r in ok) else 0.0
            p95 = percentile([r.latency_ms for r in ok], 0.95)
            print(f"    Avg latency: {avg_lat:.0f}ms  p95: {p95:.0f}ms")
            print(f"    Total tokens: {total_tok}")
            print(f"    Avg TPS: {avg_tps:.1f}")

        all_out[lvl] = {
            "total_time_s": total_s,
            "results": [asdict(r) for r in results],
        }
        await asyncio.sleep(0.5)

    return all_out


async def run_rpm_tpm_window(
    client: LLMClient,
    duration_s: int = 30,
    concurrency: int = 5,
    max_tokens: int = 20,
) -> Tuple[List[TestResult], float, float]:
    """
    Generate load for duration_s with a fixed concurrency worker pool.
    This approximates throughput under steady conditions.
    """
    print(f"\n{'='*60}\nRPM/TPM Window Test: duration={duration_s}s, concurrency={concurrency}\n{'='*60}")

    stop_at = time.perf_counter() + duration_s
    results: List[TestResult] = []
    lock = asyncio.Lock()
    counter = 0

    async def worker(wid: int):
        nonlocal counter
        while time.perf_counter() < stop_at:
            i = None
            async with lock:
                counter += 1
                i = counter
            r = await client.chat(f"Say 'ok'. Request #{i}", max_tokens=max_tokens)
            results.append(r)
            # small pacing to avoid ultra-bursting; adjust if you want "max fire"
            await asyncio.sleep(0.05 if r.success else 0.2)

    start_wall = time.perf_counter()
    tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start_wall

    ok = [r for r in results if r.success]
    total_tokens = sum(r.tokens_total for r in ok)

    rpm = (len(ok) / (elapsed / 60.0)) if elapsed > 0 else 0.0
    tpm = (total_tokens / (elapsed / 60.0)) if elapsed > 0 else 0.0

    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Total requests: {len(results)}  Successful: {len(ok)}")
    print(f"  RPM: {rpm:.1f}   TPM: {tpm:.1f}   Total tokens: {total_tokens}")

    # Show top errors quickly
    fails = [r for r in results if not r.success]
    if fails:
        tops = top_errors(fails, k=3)
        for e in tops:
            print(f"  TopErr: status={e['status_code']} count={e['count']} sample={e['sample'][:120]}")
    return results, rpm, tpm


async def warmup(client: LLMClient, n: int = 3):
    print(f"\nWarming up... ({n} requests)")
    for i in range(n):
        _ = await client.chat("Say 'warmup'.", max_tokens=10)
        await asyncio.sleep(0.2)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_levels(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


async def main_async(args):
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise SystemExit(f"ERROR: {API_KEY_ENV} not set!")

    ensure_dir(args.outdir)

    print(f"{'='*70}\nLLM BENCH (ASYNC)\n{'='*70}")
    print(f"API:   {BASE_URL}")
    print(f"Model: {args.model}")
    print(f"Key:   {api_key[:12]}...")

    output: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "api": BASE_URL,
        "model": args.model,
        "args": vars(args),
        "analyses": [],
        "raw": {},
        "final_metrics": {},
    }

    async with LLMClient(
        api_key=api_key,
        model=args.model,
        timeout_s=args.timeout,
        max_connections=args.max_connections,
        keepalive_connections=args.keepalive_connections,
    ) as client:

        await warmup(client, n=args.warmup)

        # 1) Latency
        latency_results = await run_latency_test(client, n=args.latency_n)
        output["raw"]["latency"] = [asdict(r) for r in latency_results]
        output["analyses"].append(summarize(latency_results, f"Latency (n={args.latency_n})"))

        # 2) Concurrency sweep
        sweep = await run_concurrency_sweep(client, levels=args.levels)
        output["raw"]["concurrency_sweep"] = sweep
        for lvl in args.levels:
            rs = [TestResult(**r) for r in sweep[lvl]["results"]]
            output["analyses"].append(summarize(rs, f"Concurrency (level={lvl})"))

        # 3) RPM/TPM window
        window_results, rpm, tpm = await run_rpm_tpm_window(
            client,
            duration_s=args.duration,
            concurrency=args.window_concurrency,
            max_tokens=args.window_max_tokens,
        )
        output["raw"]["rpm_tpm_window"] = [asdict(r) for r in window_results]
        output["final_metrics"] = {"rpm": rpm, "tpm": tpm}
        output["analyses"].append(summarize(window_results, f"RPM/TPM Window (c={args.window_concurrency}, {args.duration}s)"))

    # Print summary
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for a in output["analyses"]:
        print(f"\n### {a['test']}")
        print(f"  Success: {a['successful']}/{a['total_requests']} ({a.get('success_rate', 0):.1f}%)")
        if "latency" in a:
            l = a["latency"]
            print(f"  Latency: avg={l['avg_ms']:.0f}ms  med={l['median_ms']:.0f}ms  p95={l['p95_ms']:.0f}ms  max={l['max_ms']:.0f}ms")
        if "tokens" in a:
            print(f"  Tokens: {a['tokens']['total']} total  avg/req={a['tokens']['avg_per_request']:.1f}")
        if "tps" in a and a["tps"]["avg"] > 0:
            print(f"  TPS:    avg={a['tps']['avg']:.1f}  max={a['tps']['max']:.1f}")
        if a.get("errors_top"):
            print(f"  Top errors:")
            for e in a["errors_top"]:
                print(f"    - status={e['status_code']} count={e['count']} sample={e['sample'][:120]}")

    print(f"\nFINAL: RPM={output['final_metrics'].get('rpm', 0):.1f}  TPM={output['final_metrics'].get('tpm', 0):.1f}")

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.outdir, f"performance_results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {json_path}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--levels", default="1,2,5,10,20,30,40,50", help="comma-separated concurrency levels")
    p.add_argument("--duration", type=int, default=30, help="rpm/tpm window duration seconds")
    p.add_argument("--outdir", default="test-results")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--latency-n", type=int, default=10)

    # rpm/tpm window config
    p.add_argument("--window-concurrency", type=int, default=5)
    p.add_argument("--window-max-tokens", type=int, default=20)

    # network/client tuning
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-connections", type=int, default=200)
    p.add_argument("--keepalive-connections", type=int, default=100)

    return p


def main():
    args = build_parser().parse_args()
    args.levels = parse_levels(args.levels)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
