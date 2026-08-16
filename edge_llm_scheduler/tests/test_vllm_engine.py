"""契约测试：VLLMEngine 对接 vLLM OpenAI API。

不需要 GPU——用 aiohttp 起一个 fake OpenAI server（本机 loopback），
验证 VLLMEngine 发出的请求格式正确、响应解析正确、错误处理正确。

这验证的是"适配层逻辑"，不是 vLLM 本体。真实 GPU 集成留待有硬件时。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from edge_llm_scheduler.backends.vllm_engine import VLLMEngine
from edge_llm_scheduler.core.types import Task


class FakeOpenAIServer:
    """最小 fake：验证 /v1/completions 请求格式。"""

    def __init__(self, respond_stream: bool = False):
        self.received_payloads = []
        self.respond_stream = respond_stream

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # 读 HTTP 请求头 + body
        data = await reader.read(65536)
        text = data.decode("utf-8", "ignore")
        if "\r\n\r\n" in text:
            body = text.split("\r\n\r\n", 1)[1]
            try:
                self.received_payloads.append(json.loads(body))
            except Exception:
                pass
        # 响应一个标准 OpenAI completions JSON
        resp = {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 1699999999,
            "model": "fake",
            "choices": [{"index": 0, "text": "Hello from fake vllm", "logprobs": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        body = json.dumps(resp).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        await writer.drain()
        writer.close()


@pytest.mark.asyncio
async def test_vllm_engine_generate_request_format():
    server = FakeOpenAIServer()
    srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]

    engine = VLLMEngine(node_id="node_x", base_url=f"http://127.0.0.1:{port}", model="test-model")
    task = Task(task_id="t1", request_id="r1", node_id="node_x")
    task._prompt_text = "what is the capital of france?"
    task._max_tokens = 32

    result = await engine.generate(task)
    srv.close()

    assert result.text == "Hello from fake vllm"
    assert result.num_tokens == 3
    # 请求体格式正确
    assert server.received_payloads, "server received no payload"
    payload = server.received_payloads[0]
    assert payload["model"] == "test-model"
    assert payload["prompt"] == "what is the capital of france?"
    assert payload["max_tokens"] == 32
    assert payload["stream"] is False


@pytest.mark.asyncio
async def test_vllm_engine_error_handling():
    """服务端返回错误 → engine 抛异常。"""

    async def error_handler(reader, writer):
        await reader.read(65536)
        body = b'{"error": {"message": "model not loaded"}}'
        writer.write(
            b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        await writer.drain()
        writer.close()

    srv = await asyncio.start_server(error_handler, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    engine = VLLMEngine(node_id="node_x", base_url=f"http://127.0.0.1:{port}")
    task = Task(task_id="t2", request_id="r2", node_id="node_x")
    task._prompt_text = "hello"

    with pytest.raises(RuntimeError):
        await engine.generate(task)
    srv.close()


@pytest.mark.asyncio
async def test_vllm_engine_connection_refused():
    """连不上 → 抛异常（明确失败，不静默）。"""
    engine = VLLMEngine(node_id="node_x", base_url="http://127.0.0.1:1")  # 无服务
    task = Task(task_id="t3", request_id="r3", node_id="node_x")
    task._prompt_text = "hello"
    with pytest.raises(Exception):
        await engine.generate(task)


@pytest.mark.asyncio
async def test_get_status():
    server = FakeOpenAIServer()
    srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    engine = VLLMEngine(node_id="node_x", base_url=f"http://127.0.0.1:{port}")
    status = await engine.get_status()
    srv.close()
    assert status["status"] == 200
    assert "node_x" in status["url"] or "127.0.0.1" in status["url"]
