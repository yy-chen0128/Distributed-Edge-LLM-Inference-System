"""TCP 传输：有线链路的简单实现。

一个调度器管理多个节点。push 把数据发到目标节点的 TCP 服务端（在节点上跑一个
简单接收端），pull 从节点拉。为框架验证提供真实网络路径（本机即可测 loopback）。
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Dict, Optional

from ..core.transport import Transport

logger = logging.getLogger(__name__)

# 简单帧协议：4 字节长度前缀 + payload
_HEADER = struct.Struct(">I")


class TCPTransport(Transport):
    """客户端视角：连接到各节点的 TCP 接收服务。

    nodes: {node_id: (host, port)}。每个节点需运行 TcpReceiver（见下）。
    """

    def __init__(self, nodes: Dict[str, tuple], bandwidth_mbps: float = 1000.0) -> None:
        self.nodes = nodes
        self._bandwidth = bandwidth_mbps
        self._writers: Dict[str, asyncio.StreamWriter] = {}

    @property
    def bandwidth_mbps(self) -> float:
        return self._bandwidth

    async def push(self, data: bytes, dst_node: str, tag: str) -> None:
        writer = await self._get_writer(dst_node)
        payload = tag.encode() + b"\x00" + data
        writer.write(_HEADER.pack(len(payload)) + payload)
        await writer.drain()

    async def pull(self, src_node: str, tag: str) -> Optional[bytes]:
        # 简单实现：push 过去再 pull 回来（对请求/响应场景）
        # 更真实的实现是节点端维护 tag->data 映射，这里用回显
        raise NotImplementedError(
            "pull needs a request/response protocol; use push with ack for now"
        )

    async def measure_bandwidth(self, dst_node: str) -> float:
        # 简化：返回配置值（真实应发探测包实测）
        return self._bandwidth

    async def estimate_transfer_time(self, data_bytes: int, dst_node: str) -> float:
        bw_bytes_per_ms = self._bandwidth * 1024 * 1024 / 1000.0
        return data_bytes / max(1.0, bw_bytes_per_ms)

    async def _get_writer(self, node_id: str) -> asyncio.StreamWriter:
        if node_id in self._writers:
            w = self._writers[node_id]
            if not w.is_closing():
                return w
        host, port = self.nodes[node_id]
        reader, writer = await asyncio.open_connection(host, port)
        self._writers[node_id] = writer
        return writer

    async def close(self) -> None:
        """关闭所有复用连接（测试/停机时调用）。"""
        for node_id, w in list(self._writers.items()):
            w.close()
            try:
                await w.wait_closed()
            except (ConnectionError, asyncio.CancelledError):
                pass
        self._writers.clear()


class TcpReceiver:
    """节点端接收服务：接受调度器的 push，按 tag 存数据。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 9000) -> None:
        self.host = host
        self.port = port
        self._store: Dict[str, bytes] = {}
        self._server = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                header = await reader.readexactly(_HEADER.size)
                (length,) = _HEADER.unpack(header)
                payload = await reader.readexactly(length)
                tag, _, data = payload.partition(b"\x00")
                self._store[tag.decode()] = data
                logger.debug(f"tcp recv tag={tag} len={len(data)}")
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
