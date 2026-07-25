import asyncio
import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
