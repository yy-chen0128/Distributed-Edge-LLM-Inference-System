"""节点侧 stage agent：一台机器一个进程，承载它分到的那几层。

这是路线图里"阶段 C：真实 worker agent"的最小落地：控制面（
`experiments/run_real_pipeline.py`）不再直接持有 runtime 对象，而是通过 TCP
把 prepare/activate/retire/prefill/decode/status 下发给各机器的 agent。

线协议（每条连接长连接复用，顺序请求-响应）：

    [4B 头长度][头 JSON][4B 载荷长度][载荷字节]

头 JSON 形如 `{"op":"prefill","id":7,...}`；响应头形如
`{"ok":true,"id":7,"compute_ms":12.3,"queue_ms":0.4,...}`，载荷是序列化后的
hidden states（末段为空）。

为什么用长连接 + 单锁：
- 长连接：decode 每步都要过一次 stage 边界，WiFi 上每步重连要多付 1 个 RTT。
- 单锁：一个进程只有一张卡/一份权重，计算必须串行；但**不同机器上的 stage 可以
  并行处理不同请求**，这正是流水线利用率（PP 的 bubble 消除）的来源。agent 会
  把等待锁的时间作为 `queue_ms` 上报，供控制面判断该节点是否成为瓶颈。
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import socketserver
import struct
import threading
import time
from typing import Any, Optional

from ..backends.hf_layered_engine import HFLayeredEngine
from ..core.types import StageAssignment

_HEADER = struct.Struct(">I")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage-agent")


def recv_exactly(sock, count: int) -> bytes:
    chunks = b""
    while len(chunks) < count:
        chunk = sock.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError("连接在读取过程中关闭")
        chunks += chunk
    return chunks


def read_frame(sock) -> tuple[dict, bytes]:
    (header_len,) = _HEADER.unpack(recv_exactly(sock, _HEADER.size))
    header = json.loads(recv_exactly(sock, header_len).decode("utf-8"))
    (payload_len,) = _HEADER.unpack(recv_exactly(sock, _HEADER.size))
    payload = recv_exactly(sock, payload_len) if payload_len else b""
    return header, payload


def write_frame(sock, header: dict, payload: bytes = b"") -> None:
    body = json.dumps(header, ensure_ascii=False).encode("utf-8")
    sock.sendall(_HEADER.pack(len(body)) + body + _HEADER.pack(len(payload)) + payload)


class EngineService:
    """把 HFLayeredEngine 包装成 agent 可调用的命令集合。"""

    def __init__(self, engine: HFLayeredEngine, loop=None) -> None:
        self.engine = engine
        self.lock = threading.Lock()
        self.loop = loop
        self.counters = {"prefill": 0, "decode": 0, "prepare": 0, "retire": 0}
        self.busy_ms = 0.0
        self.queue_ms = 0.0

    # ---------------------------------------------------------------- 控制面

    def prepare_epoch(self, epoch: int, layer_range: list[int], model_source: str) -> dict:
        started = time.perf_counter()
        with self.lock:
            self.queue_ms += (time.perf_counter() - started) * 1000.0
            result = self._run(
                self.engine.prepare_epoch(
                    StageAssignment(epoch, self.engine.node_id, (int(layer_range[0]), int(layer_range[1]))),
                    model_source=model_source,
                )
            )
        self.counters["prepare"] += 1
        stage = self.engine._active_stage()
        return {
            "node_id": result.node_id,
            "epoch": result.pipeline_epoch,
            "layer_range": list(result.layer_range),
            "param_bytes": stage.param_bytes if stage else 0,
        }

    def activate_epoch(self, epoch: int) -> dict:
        self._run(self.engine.activate_epoch(epoch))
        return {"active_epoch": self.engine.active_epoch}

    def retire_epoch(self, epoch: int) -> dict:
        self._run(self.engine.retire_epoch(epoch))
        self.counters["retire"] += 1
        return {"retired": epoch, "active_epoch": self.engine.active_epoch}

    def status(self) -> dict:
        payload = self.engine.status()
        payload["counters"] = dict(self.counters)
        payload["busy_ms"] = round(self.busy_ms, 2)
        payload["queue_ms"] = round(self.queue_ms, 2)
        return payload

    # ---------------------------------------------------------------- 数据面

    def prefill(self, request_id: str, token_ids: Optional[list[int]],
                hidden: Any, hidden_meta: Optional[dict]) -> dict:
        started = time.perf_counter()
        with self.lock:
            queued = (time.perf_counter() - started) * 1000.0
            self.queue_ms += queued
            t0 = time.perf_counter()
            out = self.engine.prefill(request_id, token_ids=token_ids, hidden=hidden)
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self.busy_ms += compute_ms
        self.counters["prefill"] += 1
        return self._pack(out, compute_ms, queued)

    def decode(self, request_id: str, token_id: Optional[int], hidden: Any,
               hidden_meta: Optional[dict]) -> dict:
        started = time.perf_counter()
        with self.lock:
            queued = (time.perf_counter() - started) * 1000.0
            self.queue_ms += queued
            t0 = time.perf_counter()
            out = self.engine.decode(request_id, token_id=token_id, hidden=hidden)
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self.busy_ms += compute_ms
        self.counters["decode"] += 1
        return self._pack(out, compute_ms, queued)

    def release_request(self, request_id: str) -> dict:
        self.engine.release_request(request_id)
        return {"released": request_id}

    # ----------------------------------------------------------------- 内部

    def _pack(self, out, compute_ms: float, queued_ms: float) -> dict:
        header = {
            "compute_ms": round(compute_ms, 3),
            "queue_ms": round(queued_ms, 3),
            "token_id": out.token_id,
            "metrics": out.metrics,
        }
        payload = b""
        if out.hidden is not None:
            payload, meta = self.engine.serialize_hidden(out.hidden)
            header["hidden_meta"] = meta
            header["hidden_bytes"] = len(payload)
        else:
            header["hidden_bytes"] = 0
        return {"header": header, "payload": payload}

    @staticmethod
    def _run(coro):
        """在 agent 进程里同步等待 engine 的 async 接口。"""
        import asyncio

        return asyncio.run(coro)


class _Handler(socketserver.BaseRequestHandler):
    @property
    def service(self) -> EngineService:
        return self.server.service  # type: ignore[attr-defined]

    def handle(self) -> None:
        sock = self.request
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # decode 每步都是小包
        except OSError:  # pragma: no cover - 平台差异，不影响功能
            pass
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        logger.info("连接来自 %s", peer)
        try:
            while True:
                header, payload = read_frame(sock)
                response_header = self._dispatch(header, payload)
                write_frame(sock, response_header["header"], response_header.get("payload", b""))
        except (ConnectionError, OSError):
            logger.info("连接结束 %s", peer)

    def _dispatch(self, header: dict, payload: bytes) -> dict:
        op = header.get("op")
        request_id = header.get("id")
        try:
            if op == "hello":
                return {"header": {"ok": True, "id": request_id, **self.service.engine.inventory()}}
            if op == "status":
                return {"header": {"ok": True, "id": request_id, **self.service.status()}}
            if op == "prepare_epoch":
                result = self.service.prepare_epoch(
                    int(header["epoch"]), list(header["layer_range"]),
                    str(header.get("model_source", "model-store")),
                )
                return {"header": {"ok": True, "id": request_id, **result}}
            if op == "activate_epoch":
                return {"header": {"ok": True, "id": request_id,
                                   **self.service.activate_epoch(int(header["epoch"]))}}
            if op == "retire_epoch":
                return {"header": {"ok": True, "id": request_id,
                                   **self.service.retire_epoch(int(header["epoch"]))}}
            if op == "release_request":
                return {"header": {"ok": True, "id": request_id,
                                   **self.service.release_request(str(header["request_id"]))}}
            if op in ("prefill", "decode"):
                hidden = None
                if payload:
                    hidden, _ = self.service.engine.deserialize_hidden(
                        payload, header.get("hidden_meta")
                    )
                if op == "prefill":
                    packed = self.service.prefill(
                        str(header["request_id"]),
                        header.get("tokens"),
                        hidden,
                        header.get("hidden_meta"),
                    )
                else:
                    packed = self.service.decode(
                        str(header["request_id"]),
                        header.get("token"),
                        hidden,
                        header.get("hidden_meta"),
                    )
                return {"header": {"ok": True, "id": request_id, **packed["header"]},
                        "payload": packed["payload"]}
            if op == "shutdown":
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return {"header": {"ok": True, "id": request_id, "shutdown": True}}
            raise ValueError(f"未知操作 {op}")
        except Exception as exc:  # 把错误回给控制面，而不是静默断开
            logger.exception("处理 %s 失败", op)
            return {"header": {"ok": False, "id": request_id,
                               "error": f"{type(exc).__name__}: {exc}"}}


class AgentServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    service: EngineService

    def __init__(self, address, handler, service: EngineService):
        self.service = service
        super().__init__(address, handler)


def build_server(host: str, port: int, engine: HFLayeredEngine) -> AgentServer:
    service = EngineService(engine)
    return AgentServer((host, port), _Handler, service)


def main() -> int:
    parser = argparse.ArgumentParser(description="PAIR-style stage agent（本项目自研节点 agent）")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default=".models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0 / auto")
    parser.add_argument("--dtype", default="auto", help="auto / float32 / float16 / bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--wire-dtype", default="float16",
                        help="链路传输 dtype：float16（默认，省一半带宽）/ bfloat16 / float32 / none")
    parser.add_argument("--threads", type=int, default=0,
                        help="本进程 torch 线程数（0=不限制）。单机多 agent 时按核数分摊，避免超订")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    if args.threads > 0:
        import torch

        torch.set_num_threads(args.threads)
        logger.info("torch 线程数限制为 %d", args.threads)
    wire = "" if args.wire_dtype.lower() in ("none", "") else args.wire_dtype
    engine = HFLayeredEngine(
        node_id=args.node_id,
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        wire_dtype=wire,
        logger=logger.info,
    )
    server = build_server(args.host, args.port, engine)
    logger.info("agent %s 监听 %s:%d device=%s dtype=%s 模型=%s",
                args.node_id, args.host, args.port, engine.device, engine.dtype, args.model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断，退出")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
