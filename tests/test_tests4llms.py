import asyncio
import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

from aiohttp import web


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tests4llms-v0.2.py"
SPEC = importlib.util.spec_from_file_location("tests4llms_v02", SCRIPT_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


class ArgumentTests(unittest.TestCase):
    def parse_args(self, *extra):
        base = [
            "--base-url", "https://example.test/v1",
            "--api-key", "test-key",
            "--model", "test-model",
        ]
        return BENCH.build_parser().parse_args(base + list(extra))

    def test_parse_concurrency_sorts_and_deduplicates(self):
        self.assertEqual(BENCH.parse_concurrency("5,1,5,2"), [1, 2, 5])

    def test_parse_concurrency_rejects_invalid_values(self):
        for value in ("", "0", "-2,5", "one,2"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                BENCH.parse_concurrency(value)

    def test_validate_rejects_invalid_numeric_options(self):
        cases = (
            ("--mode", "burst", "--rounds", "0"),
            ("--mode", "steady", "--duration", "0"),
            ("--max-tokens", "0"),
            ("--timeout", "0"),
            ("--warmup", "-1"),
            ("--output-dir", ""),
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(SystemExit):
                args = self.parse_args(*options)
                BENCH.validate_args(args)

    def test_provider_presets_choose_compatible_token_parameter(self):
        openai = BENCH.build_parser().parse_args(["--provider", "openai"])
        BENCH.apply_provider(openai)
        BENCH.validate_args(openai)
        self.assertEqual(openai.token_param, "max_completion_tokens")

        anthropic = BENCH.build_parser().parse_args(
            ["--provider", "anthropic"]
        )
        BENCH.apply_provider(anthropic)
        BENCH.validate_args(anthropic)
        self.assertEqual(anthropic.base_url, "https://api.anthropic.com/v1/")
        self.assertEqual(anthropic.model, "claude-opus-5")
        self.assertEqual(anthropic.token_param, "max_tokens")

        stepfun = BENCH.build_parser().parse_args(["--provider", "stepfun"])
        BENCH.apply_provider(stepfun)
        BENCH.validate_args(stepfun)
        self.assertEqual(stepfun.model, "step-3.7-flash")
        self.assertEqual(stepfun.token_param, "max_tokens")

    def test_manual_configuration_defaults_to_max_tokens(self):
        args = self.parse_args()
        BENCH.validate_args(args)
        self.assertEqual(args.token_param, "max_tokens")

    def test_stream_usage_can_be_disabled_explicitly(self):
        self.assertTrue(self.parse_args().stream_usage)
        args = self.parse_args("--stream", "--no-stream-usage")
        self.assertFalse(args.stream_usage)


class RunnerGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_burst_rejects_zero_work(self):
        with self.assertRaises(ValueError):
            await BENCH.run_burst(None, 0, 1, 1, "prompt")
        with self.assertRaises(ValueError):
            await BENCH.run_burst(None, 1, 0, 1, "prompt")

    async def test_steady_rejects_zero_work(self):
        with self.assertRaises(ValueError):
            await BENCH.run_steady(None, 0, 1, 1, "prompt")
        with self.assertRaises(ValueError):
            await BENCH.run_steady(None, 1, 0, 1, "prompt")


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []

    async def asyncTearDown(self):
        for runner in reversed(self.runners):
            await runner.cleanup()

    async def start_server(self, handler):
        app = web.Application()
        app.router.add_post("/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.runners.append(runner)
        port = site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def test_non_stream_latency_includes_delayed_success_body(self):
        async def handler(request):
            response = web.StreamResponse(
                headers={"Content-Type": "application/json"},
            )
            await response.prepare(request)
            await asyncio.sleep(0.1)
            await response.write(json.dumps({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 10,
                          "total_tokens": 12},
            }).encode())
            await response.write_eof()
            return response

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model"
        ) as client:
            result = await client.chat("hello")

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.latency_ms, 90)
        self.assertLessEqual(result.per_request_tok_per_sec, 10 / 0.09)
        self.assertAlmostEqual(
            result.per_request_tok_per_sec,
            10 / (result.latency_ms / 1000),
        )

    async def test_non_stream_latency_includes_delayed_error_body(self):
        async def handler(request):
            response = web.StreamResponse(status=429)
            await response.prepare(request)
            await asyncio.sleep(0.1)
            await response.write(b"rate limited")
            await response.write_eof()
            return response

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model"
        ) as client:
            result = await client.chat("hello")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 429)
        self.assertEqual(result.error, "rate limited")
        self.assertGreaterEqual(result.latency_ms, 90)

    async def test_strict_stream_server_supports_omitting_stream_options(self):
        received = []

        async def handler(request):
            payload = await request.json()
            received.append(payload)
            if "stream_options" in payload:
                return web.json_response(
                    {"error": "unknown field stream_options"}, status=400,
                )
            return web.Response(
                text='data: {"choices":[{"delta":{"content":"ok"},'
                     '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                content_type="text/event-stream",
            )

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model", stream=True
        ) as client:
            default_result = await client.chat("hello")

        self.assertFalse(default_result.success)
        self.assertEqual(default_result.status_code, 400)
        self.assertEqual(received[0]["stream_options"], {"include_usage": True})

        async with BENCH.LLMClient(
            base_url, "test-key", "test-model", stream=True,
            stream_usage=False,
        ) as client:
            result = await client.chat("hello")

        self.assertTrue(result.success)
        self.assertNotIn("stream_options", received[1])

    async def test_warmup_uses_benchmark_prompt_and_output_limit(self):
        received = []

        async def handler(request):
            received.append(await request.json())
            return web.json_response({
                "choices": [{"message": {"content": "ok"}}],
            })

        base_url = await self.start_server(handler)
        for token_param in ("max_tokens", "max_completion_tokens"):
            with self.subTest(token_param=token_param):
                received.clear()
                async with BENCH.LLMClient(
                    base_url, "test-key", "test-model",
                    token_param=token_param,
                ) as client:
                    results = await BENCH.warmup(
                        client, 2, 512, "actual benchmark prompt",
                    )
                self.assertEqual(len(results), 2)
                self.assertTrue(all(result.success for result in results))
                self.assertEqual(len(received), 2)
                for payload in received:
                    self.assertEqual(payload[token_param], 512)
                    self.assertEqual(payload["messages"], [{
                        "role": "user", "content": "actual benchmark prompt",
                    }])

    async def test_missing_usage_is_unknown_and_explicit_zero_is_preserved(self):
        usage = None

        async def handler(request):
            payload = await request.json()
            if payload["stream"]:
                chunk = {"choices": [{"delta": {"content": "ok"},
                                      "finish_reason": "stop"}]}
                if usage is not None:
                    chunk["usage"] = usage
                return web.Response(
                    text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                    content_type="text/event-stream",
                )
            body = {"choices": [{"message": {"content": "ok"}}]}
            if usage is not None:
                body["usage"] = usage
            return web.json_response(body)

        base_url = await self.start_server(handler)
        for stream in (False, True):
            for has_usage in (False, True):
                with self.subTest(stream=stream, has_usage=has_usage):
                    usage = ({"prompt_tokens": 0, "completion_tokens": 0,
                              "total_tokens": 0} if has_usage else None)
                    async with BENCH.LLMClient(
                        base_url, "test-key", "test-model", stream=stream,
                    ) as client:
                        result = await client.chat("hello")
                    self.assertTrue(result.success)
                    expected = 0 if has_usage else None
                    self.assertEqual(result.tokens_prompt, expected)
                    self.assertEqual(result.tokens_completion, expected)
                    self.assertEqual(result.tokens_total, expected)
                    self.assertEqual(result.per_request_tok_per_sec, expected)

    async def test_run_excludes_warmup_and_records_reproducible_config(self):
        received = []

        async def handler(request):
            received.append(await request.json())
            if len(received) == 1:
                return web.json_response({"error": "warmup failure"}, status=429)
            completion_tokens = 100 if len(received) == 2 else 3
            return web.json_response({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2,
                          "completion_tokens": completion_tokens,
                          "total_tokens": 2 + completion_tokens},
            })

        base_url = await self.start_server(handler)
        prompt = "alpha | beta\nsecond line"
        with tempfile.TemporaryDirectory() as output_dir:
            args = BENCH.build_parser().parse_args([
                "--provider", "openai", "--base-url", base_url,
                "--api-key", "test-key", "--model", "local-test-model",
                "--concurrency", "1", "--rounds", "1", "--warmup", "2",
                "--max-tokens", "512", "--prompt", prompt,
                "--timeout", "7", "--output-dir", output_dir,
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                await BENCH.run(args)
            output = json.loads(
                (Path(output_dir) / "result.json").read_text(encoding="utf-8")
            )
            report = (Path(output_dir) / "report.md").read_text(encoding="utf-8")

        self.assertEqual(len(received), 3)
        for payload in received:
            self.assertEqual(payload["max_completion_tokens"], 512)
            self.assertEqual(payload["messages"][0]["content"], prompt)
        self.assertEqual(len(output["raw"]["c1"]), 1)
        summary = output["summaries"][0]
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["tokens"]["completion_total"], 3)
        self.assertGreater(summary["elapsed_s"], 0)
        config = output["config"]
        for key, expected in {
            "provider": "openai", "prompt": prompt, "warmup": 2,
            "warmup_successful": 1, "timeout": 7,
            "python_version": sys.version.split()[0],
            "aiohttp_version": BENCH.aiohttp.__version__,
        }.items():
            self.assertEqual(config[key], expected)
        self.assertNotIn("api_key", config)
        self.assertNotIn("test-key", report)
        for label in ("provider", "prompt", "warmup", "timeout", "python", "aiohttp"):
            self.assertIn(label, report.lower())
        self.assertIn("local-test-model", report)
        prompt_row = next(line for line in report.splitlines()
                          if line.lower().startswith("| prompt "))
        for part in ("alpha", "beta", "second line"):
            self.assertIn(part, prompt_row)
        self.assertEqual(len(re.findall(r"(?<!\\)\|", prompt_row)), 3)

    async def test_steady_elapsed_includes_draining_in_flight_request(self):
        async def handler(request):
            await asyncio.sleep(0.08)
            return web.json_response({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            })

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model",
        ) as client:
            results, elapsed = await BENCH.run_steady(
                client, 1, 0.03, 10, "hello",
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertGreaterEqual(elapsed, 0.07)
        summary = BENCH.summarize_level(results, 1, elapsed, mode="steady")
        self.assertEqual(summary["elapsed_s"], elapsed)
        self.assertAlmostEqual(summary["req_per_sec"], 1 / elapsed)

    async def test_run_handles_disabled_stream_usage_and_unknown_token_counts(self):
        received = []

        async def handler(request):
            payload = await request.json()
            received.append(payload)
            if "stream_options" in payload:
                return web.json_response({"error": "unknown field"}, status=400)
            if payload["stream"]:
                return web.Response(
                    text='data: {"choices":[{"delta":{"content":"ok"},'
                         '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                    content_type="text/event-stream",
                )
            return web.json_response({
                "choices": [{"message": {"content": "ok"}}],
            })

        base_url = await self.start_server(handler)
        for stream in (False, True):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as out:
                options = [
                    "--base-url", base_url, "--api-key", "test-key",
                    "--model", "test-model", "--concurrency", "1",
                    "--rounds", "1", "--warmup", "0", "--no-stream-usage",
                    "--output-dir", out,
                ]
                if stream:
                    options.append("--stream")
                args = BENCH.build_parser().parse_args(options)
                with contextlib.redirect_stdout(io.StringIO()):
                    await BENCH.run(args)
                output = json.loads(
                    (Path(out) / "result.json").read_text(encoding="utf-8")
                )
                report = (Path(out) / "report.md").read_text(encoding="utf-8")
                self.assertEqual(output["summaries"][0]["successful"], 1)
                self.assertFalse(output["config"]["stream_usage"])
                self.assertIsNone(output["raw"]["c1"][0]["tokens_completion"])
                self.assertIsNone(output["summaries"][0]["completion_tok_per_sec"])
                self.assertIn("n/a", report)
                self.assertEqual(received[-1]["stream"], stream)
                self.assertNotIn("stream_options", received[-1])

    async def test_connection_limit_and_token_parameter_are_applied(self):
        received = {}

        async def handler(request):
            received.update(await request.json())
            return web.json_response({
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            })

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url,
            "test-key",
            "test-model",
            max_connections=513,
            token_param="max_completion_tokens",
        ) as client:
            result = await client.chat("hello", max_tokens=7)
            self.assertEqual(client.connector.limit, 513)
            self.assertEqual(client.connector.limit_per_host, 513)

        self.assertTrue(result.success)
        self.assertEqual(received["max_completion_tokens"], 7)
        self.assertNotIn("max_tokens", received)

    async def test_empty_http_200_stream_is_failure(self):
        async def handler(request):
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write_eof()
            return response

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model", stream=True
        ) as client:
            result = await client.chat("hello")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            result.error, "stream contained no completion chunks"
        )

    async def test_empty_http_200_non_stream_response_is_failure(self):
        async def handler(request):
            return web.json_response({})

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model"
        ) as client:
            result = await client.chat("hello")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            result.error, "response contained no completion choices"
        )

    async def test_done_marker_finishes_without_waiting_for_socket_close(self):
        release_handler = asyncio.Event()

        async def handler(request):
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"delta":{"content":"ok"},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":1,'
                b'"completion_tokens":1,"total_tokens":2}}\n\n'
                b"data: [DONE]\n\n"
            )
            await release_handler.wait()
            return response

        base_url = await self.start_server(handler)
        try:
            async with BENCH.LLMClient(
                base_url,
                "test-key",
                "test-model",
                timeout_s=0.2,
                stream=True,
            ) as client:
                result = await client.chat("hello")
        finally:
            release_handler.set()

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertGreater(result.ttft_ms, 0)
        self.assertEqual(result.tokens_completion, 1)

    async def test_reasoning_content_sets_ttft(self):
        async def handler(request):
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"delta":{"content":"",'
                b'"role":"assistant"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{"role":"assistant",'
                b'"reasoning_content":"thinking"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{"content":""},'
                b'"finish_reason":"length"}],"usage":{"prompt_tokens":1,'
                b'"completion_tokens":2,"total_tokens":3}}\n\n'
                b"data: [DONE]\n\n"
            )
            await response.write_eof()
            return response

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model", stream=True
        ) as client:
            result = await client.chat("hello")

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertGreater(result.ttft_ms, 0)
        self.assertEqual(result.tokens_completion, 2)

    async def test_truncated_stream_is_failure(self):
        async def handler(request):
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"delta":{"content":"partial"},'
                b'"finish_reason":null}]}\n\n'
            )
            await response.write_eof()
            return response

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model", stream=True
        ) as client:
            result = await client.chat("hello")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.error, "stream ended before completion")

    async def test_finish_reason_allows_compatible_stream_without_done(self):
        async def handler(request):
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"delta":{"content":"ok"},'
                b'"finish_reason":"stop"}]}\n\n'
            )
            await response.write_eof()
            return response

        base_url = await self.start_server(handler)
        async with BENCH.LLMClient(
            base_url, "test-key", "test-model", stream=True
        ) as client:
            result = await client.chat("hello")

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)


class MetricsAndReportTests(unittest.TestCase):
    def config(self, mode="burst"):
        return {
            "model": "test-model", "base_url": "http://local.test",
            "provider": "openai", "mode": mode,
            "timestamp": "2026-10-23T12:00:00", "concurrency": "1",
            "connection_limit": 1, "rounds": 1, "duration": 10,
            "max_tokens": 512, "token_param": "max_completion_tokens",
            "stream": False, "stream_usage": True, "prompt": "hello",
            "warmup": 2, "warmup_successful": 2, "timeout": 7,
            "python_version": sys.version.split()[0],
            "aiohttp_version": BENCH.aiohttp.__version__,
        }

    def test_unknown_usage_is_not_reported_as_zero(self):
        result = BENCH.BenchResult(True, 200, 100)
        summary = BENCH.summarize_level([result], 1, 0.1)
        self.assertIsNone(summary["completion_tok_per_sec"])
        self.assertIsNone(summary["per_request_tok_per_sec"])
        for key in ("prompt_total", "completion_total", "avg_completion"):
            self.assertIsNone(summary["tokens"][key])
        report = BENCH.generate_report(self.config(), [summary])
        row = next(line for line in report.splitlines()
                   if line.startswith("| 1 |"))
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        self.assertEqual(cells[-2:], ["n/a", "n/a"])

    def test_known_zero_usage_keeps_zero_statistics_and_report(self):
        result = BENCH.BenchResult(
            True, 200, 100, tokens_prompt=0, tokens_completion=0,
            tokens_total=0, per_request_tok_per_sec=0,
        )
        summary = BENCH.summarize_level([result], 1, 0.1)
        self.assertEqual(summary["completion_tok_per_sec"], 0)
        self.assertEqual(summary["per_request_tok_per_sec"], 0)
        for key in ("prompt_total", "completion_total", "avg_completion"):
            self.assertEqual(summary["tokens"][key], 0)
        report = BENCH.generate_report(self.config(), [summary])
        row = next(line for line in report.splitlines()
                   if line.startswith("| 1 |"))
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        self.assertEqual(cells[-2:], ["0.0", "0.0"])

    def test_partial_usage_does_not_present_partial_sum_as_total(self):
        known = BENCH.BenchResult(
            True, 200, 100, tokens_prompt=2, tokens_completion=3,
            tokens_total=5, per_request_tok_per_sec=30,
        )
        missing_completion = BENCH.BenchResult(
            True, 200, 100, tokens_prompt=4,
        )
        summary = BENCH.summarize_level([known, missing_completion], 1, 0.2)
        self.assertEqual(summary["tokens"]["prompt_total"], 6)
        self.assertIsNone(summary["tokens"]["completion_total"])
        self.assertIsNone(summary["tokens"]["avg_completion"])
        self.assertIsNone(summary["completion_tok_per_sec"])
        self.assertIsNone(summary["per_request_tok_per_sec"])

    def test_failed_level_still_records_elapsed_time(self):
        result = BENCH.BenchResult(False, 429, 100, error="rate limited")
        summary = BENCH.summarize_level([result], 1, 0.125)
        self.assertEqual(summary["elapsed_s"], 0.125)

    def test_steady_report_explains_dispatch_window_and_drain(self):
        result = BENCH.BenchResult(True, 200, 11000)
        summary = BENCH.summarize_level([result], 1, 11, mode="steady")
        report = BENCH.generate_report(self.config("steady"), [summary])
        self.assertIn("dispatch window", report.lower())
        self.assertIn("drain", report.lower())
        self.assertIn("elapsed", report.lower())


if __name__ == "__main__":
    unittest.main()
