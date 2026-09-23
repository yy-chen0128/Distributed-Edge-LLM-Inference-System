"""入口网关：档0（气泡重启）+ 档1（拼接提示词重放）。

**为什么需要它**：vLLM 不提供"停接纳 / drain / 重建后把在途请求救回来"的语义，
档0 与档1 全部落在这个网关上：

1. **累积**：边转发边记录"本次请求已经产出的内容"（这是重放的原料）；
2. **判故障**：下游连接失败 / 流中断 / 超时 → 标记流水线不可用；
3. **停接纳 + drain**：新请求直接 503（或排队）；已在途的交给第 4 步；
4. **重放**：流水线重建完成（`POST /admin/resume`，或上游 `/v1/models` 重新可用）后，
   把 `[原 prompt + 已生成内容]` 作为新的输入**重新提交**，
   **只把新增的部分转发给客户端**——客户端那条 SSE 流全程不断，只感到一次卡顿；
5. **记录重放保真度**：重放出来的前缀与"已经发给客户端的内容"是否逐字一致。
   这正好把"文本重放要重新分词、不保证恒等"这个坑变成一个**可测量的指标**
   （见 `docs/deploy/vllm-lmcache-4node-plan.md` §5.3）。

用法：
    python -m edge_llm_scheduler.gateway.vllm_gateway \
        --upstream http://127.0.0.1:8000 --port 8100

    # 节点离开后：先停接纳，等流水线重建好，再放开
    curl -X POST localhost:8100/admin/drain
    curl -X POST localhost:8100/admin/resume
    curl localhost:8100/admin/status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

try:  # 重活依赖：只在真正起网关时才需要，纯函数（拼接/切分）单独可测
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    _WEB_DEPS_ERROR: Optional[str] = None
except ImportError as _exc:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    FastAPI = Request = JSONResponse = StreamingResponse = None  # type: ignore[assignment]
    _WEB_DEPS_ERROR = (
        "网关需要 fastapi/uvicorn/httpx：\n"
        "  ~/venvs/pair/bin/pip install fastapi uvicorn httpx\n"
        f"（原始错误：{_exc}）"
    )


def require_web_deps() -> None:
    if _WEB_DEPS_ERROR:
        raise SystemExit(_WEB_DEPS_ERROR)


# --------------------------------------------------------------------------- 状态


@dataclass
class PipelineState:
    """流水线可用性 + 重建信号。档0 的"停接纳 / 恢复接纳"就是这里的两个动作。"""

    available: bool = True
    reason: str = "initial"
    changed_at: float = field(default_factory=time.time)
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)

    # 计数（给实验用）
    admitted: int = 0
    rejected: int = 0
    failed: int = 0
    replayed: int = 0
    replay_exact: int = 0
    replay_diverged: int = 0

    def __post_init__(self) -> None:
        if self.available:
            self.resume_event.set()

    def mark_down(self, reason: str) -> None:
        if self.available:
            self.available = False
            self.reason = reason
            self.changed_at = time.time()
            self.resume_event.clear()

    def mark_up(self, reason: str = "resumed") -> None:
        self.available = True
        self.reason = reason
        self.changed_at = time.time()
        self.resume_event.set()

    def snapshot(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "changed_at": self.changed_at,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "failed": self.failed,
            "replayed": self.replayed,
            "replay_exact": self.replay_exact,
            "replay_diverged": self.replay_diverged,
        }


# ------------------------------------------------------------------- 重放体构造


def build_replay_body(endpoint: str, original: dict, generated: str) -> dict:
    """把"已生成的内容"拼回输入，构造重放用的请求体。

    - `/v1/completions`：prompt 是字符串 → `prompt + generated`
    - `/v1/chat/completions`：把 generated 作为一条 assistant 消息追加
    """
    body = json.loads(json.dumps(original))          # 深拷贝，别改原请求
    body.pop("stream_options", None)
    if endpoint.endswith("/chat/completions"):
        messages = list(body.get("messages") or [])
        messages.append({"role": "assistant", "content": generated})
        body["messages"] = messages
    else:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):                  # 多 prompt：只支持首个
            prompt = prompt[0] if prompt else ""
        body["prompt"] = f"{prompt}{generated}"
    return body


def extract_delta(endpoint: str, chunk: dict) -> str:
    """从流式分片里取出"这一片新增的文本"。"""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    if endpoint.endswith("/chat/completions"):
        delta = first.get("delta") or {}
        return delta.get("content") or ""
    return first.get("text") or ""


def extract_full(endpoint: str, payload: dict) -> str:
    """从非流式响应里取出完整文本。"""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    if endpoint.endswith("/chat/completions"):
        message = first.get("message") or {}
        return message.get("content") or ""
    return first.get("text") or ""


def skip_already_sent(replay_text: str, already: str) -> tuple[str, bool]:
    """从重放输出里去掉已经发给客户端的那段。

    返回 (要转发的新增部分, 前缀是否逐字一致)。`False` 表示重放的前缀与已发送内容
    不一致（采样/分词导致），此时仍然转发新增部分，但计数 divergence 供分析。
    """
    if not already:
        return replay_text, True
    if replay_text.startswith(already):
        return replay_text[len(already):], True
    # 前缀不一致：尽量找到最长公共前缀，从那里往后发（保守做法，宁可多给一点）
    limit = min(len(replay_text), len(already))
    same = 0
    while same < limit and replay_text[same] == already[same]:
        same += 1
    return replay_text[same:], False


# --------------------------------------------------------------------- 网关主体


class Gateway:
    def __init__(self, upstream: str, state: Optional[PipelineState] = None,
                 max_replays: int = 2, resume_timeout_s: float = 600.0,
                 drain_on_error: bool = True, stall_timeout_s: float = 30.0) -> None:
        require_web_deps()
        self.upstream = upstream.rstrip("/")
        self.state = state or PipelineState()
        self.max_replays = max_replays
        self.resume_timeout_s = resume_timeout_s
        self.drain_on_error = drain_on_error
        # 停滞检测：上游进程被杀时，SSE 连接可能既不报错也不结束（实测会永久挂住），
        # 所以必须给"多久没收到任何字节"设一个上限，超时即当作故障 → 触发档1 恢复。
        self.stall_timeout_s = stall_timeout_s
        self.app = self._build_app()

    def _client(self, streaming: bool):
        """按用途构造客户端：流式请求用停滞超时，非流式用普通超时。"""
        if not streaming:
            return httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        return httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=self.stall_timeout_s,
                                  write=30.0, pool=10.0))

    # ---------------------------------------------------------------- 路由
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="distributed-edge-llm gateway")

        @app.get("/admin/status")
        async def status() -> JSONResponse:
            return JSONResponse({**self.state.snapshot(), "upstream": self.upstream})

        @app.post("/admin/drain")
        async def drain(request: Request) -> JSONResponse:
            reason = "manual"
            try:
                payload = await request.json()
                reason = str(payload.get("reason") or reason)
            except Exception:  # noqa: BLE001 - body 可选
                pass
            self.state.mark_down(reason)
            return JSONResponse({"ok": True, **self.state.snapshot()})

        @app.post("/admin/resume")
        async def resume() -> JSONResponse:
            self.state.mark_up("manual resume")
            return JSONResponse({"ok": True, **self.state.snapshot()})

        @app.get("/v1/models")
        async def models() -> JSONResponse:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(f"{self.upstream}/v1/models")
                return JSONResponse(resp.json(), status_code=resp.status_code)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse({"error": f"upstream unreachable: {exc}"},
                                    status_code=502)

        @app.post("/v1/{rest:path}")
        async def proxy(rest: str, request: Request) -> Any:
            endpoint = f"/v1/{rest}"
            body = await request.json()

            # 档0 的"停接纳"
            if not self.state.available:
                self.state.rejected += 1
                return JSONResponse(
                    {"error": "pipeline unavailable", "reason": self.state.reason},
                    status_code=503)

            self.state.admitted += 1
            streaming = bool(body.get("stream"))
            if streaming:
                return StreamingResponse(
                    self._stream_with_recovery(endpoint, body),
                    media_type="text/event-stream")
            return await self._nonstream_with_recovery(endpoint, body)

        return app

    # ------------------------------------------------------------ 失败处理
    async def _wait_for_pipeline(self, label: str) -> bool:
        """等流水线重建完成（admin resume），或等上游重新可用。超时返回 False。"""
        deadline = time.time() + self.resume_timeout_s
        while time.time() < deadline:
            if self.state.available:
                return True
            # 上游可能已自行恢复（例如 Ray/vLLM 重建完成）
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self.upstream}/v1/models")
                if resp.status_code == 200:
                    self.state.mark_up("upstream recovered")
                    return True
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(self.state.resume_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
        print(f"[gateway] {label}: resume timeout after {self.resume_timeout_s}s")
        return False

    # -------------------------------------------------------------- 流式
    async def _stream_with_recovery(self, endpoint: str, body: dict) -> AsyncIterator[bytes]:
        sent = ""
        attempt = 0
        current = body
        while True:
            try:
                async for chunk_text, raw in self._stream_once(endpoint, current):
                    sent += chunk_text
                    yield raw
                return
            except Exception as exc:  # noqa: BLE001 - 任何下游异常都走恢复
                self.state.failed += 1
                print(f"[gateway] stream failed (attempt {attempt}): {type(exc).__name__}: {exc}")
                if self.drain_on_error:
                    self.state.mark_down(f"{type(exc).__name__}: {exc}")
                attempt += 1
                if attempt > self.max_replays:
                    yield self._sse_error(f"recovery exhausted: {exc}")
                    return
                if not await self._wait_for_pipeline("stream"):
                    yield self._sse_error("pipeline did not come back in time")
                    return

                # 档1：把已生成内容拼回输入，重放，只转发新增部分
                current = build_replay_body(endpoint, body, sent)
                self.state.replayed += 1
                sent_len = len(sent)
                replay_all = await self._collect_stream_text(endpoint, current)
                new_text, exact = skip_already_sent(replay_all, sent)
                if exact:
                    self.state.replay_exact += 1
                else:
                    self.state.replay_diverged += 1
                print(f"[gateway] replay #{self.state.replayed}: sent={sent_len} chars, "
                      f"replay={len(replay_all)} chars, exact={exact}, "
                      f"forwarding {len(new_text)} new chars")
                # 以重放结果为准继续（若用户设置 stream=False 也无所谓，这里统一走流式）
                for piece in self._chunk_text_as_sse(endpoint, new_text):
                    sent += piece[0]
                    yield piece[1]
                return

    async def _stream_once(self, endpoint: str, body: dict) -> AsyncIterator[tuple[str, bytes]]:
        async with self._client(streaming=True) as client:
            async with client.stream("POST", f"{self.upstream}{endpoint}", json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:200]
                    raise RuntimeError(f"upstream HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    raw = (line + "\n\n").encode("utf-8")
                    delta = ""
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload and payload != "[DONE]":
                            try:
                                delta = extract_delta(endpoint, json.loads(payload))
                            except json.JSONDecodeError:
                                delta = ""
                    yield delta, raw

    async def _collect_stream_text(self, endpoint: str, body: dict) -> str:
        text = ""
        async for delta, _ in self._stream_once(endpoint, body):
            text += delta
        return text

    @staticmethod
    def _chunk_text_as_sse(endpoint: str, text: str) -> list[tuple[str, bytes]]:
        """把一段文本重新包装成 OpenAI 兼容的流式分片（客户端无感）。"""
        out: list[tuple[str, bytes]] = []
        for i in range(0, len(text), 64):
            piece = text[i:i + 64]
            if endpoint.endswith("/chat/completions"):
                payload = {"choices": [{"delta": {"content": piece}, "index": 0}]}
            else:
                payload = {"choices": [{"text": piece, "index": 0}]}
            out.append((piece, f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")))
        out.append(("", b"data: [DONE]\n\n"))
        return out

    @staticmethod
    def _sse_error(message: str) -> bytes:
        payload = {"error": {"message": message, "type": "gateway_recovery_failed"}}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    # ------------------------------------------------------------ 非流式
    async def _nonstream_with_recovery(self, endpoint: str, body: dict) -> JSONResponse:
        attempt = 0
        current = body
        while True:
            try:
                async with self._client(streaming=False) as client:
                    resp = await client.post(f"{self.upstream}{endpoint}", json=current)
                if resp.status_code >= 400:
                    raise RuntimeError(f"upstream HTTP {resp.status_code}")
                return JSONResponse(resp.json(), status_code=resp.status_code)
            except Exception as exc:  # noqa: BLE001
                self.state.failed += 1
                if self.drain_on_error:
                    self.state.mark_down(str(exc))
                attempt += 1
                if attempt > self.max_replays or not await self._wait_for_pipeline("nonstream"):
                    return JSONResponse({"error": f"recovery failed: {exc}"}, status_code=502)
                # 非流式没有"已发送内容"，直接原样重试
                current = body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max-replays", type=int, default=2)
    parser.add_argument("--resume-timeout", type=float, default=600.0)
    parser.add_argument("--stall-timeout", type=float, default=30.0,
                        help="秒：多久收不到任何字节就当作上游故障（必须设，否则会被挂住）")
    args = parser.parse_args()

    import uvicorn

    require_web_deps()
    gateway = Gateway(args.upstream, max_replays=args.max_replays,
                      resume_timeout_s=args.resume_timeout,
                      stall_timeout_s=args.stall_timeout)
    print(f"[gateway] upstream={args.upstream} host={args.host} port={args.port} "
          f"stall_timeout={args.stall_timeout}s")
    uvicorn.run(gateway.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
