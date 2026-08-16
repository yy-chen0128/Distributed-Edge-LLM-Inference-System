"""WiFi 传输：无线链路的抽象占位。

有线(TCP)与无线(WiFi)的差异在真实链路参数：
- 带宽低（几十 MB/s vs 千兆）
- 时延波动大、有丢包
- 可能断连/重连

本文件提供 WiFiTransport：复用 TCP 帧协议，但强制无线参数（低带宽/高时延/丢包），
并通过链路状态接口暴露波动特性。真实无线实现（WiGig 等）后续补充。
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Dict, Optional

from .tcp_transport import TCPTransport

logger = logging.getLogger(__name__)


class WiFiTransport(TCPTransport):
    """在 TCP 之上模拟无线链路特性。"""

    def __init__(
        self,
        nodes: Dict[str, tuple],
        bandwidth_mbps: float = 30.0,       # WiFi 典型几十 MB/s
        latency_ms: float = 20.0,
        loss_rate: float = 0.02,            # 2% 丢包
    ) -> None:
        super().__init__(nodes, bandwidth_mbps=bandwidth_mbps)
        self.latency_ms = latency_ms
        self.loss_rate = loss_rate

    async def push(self, data: bytes, dst_node: str, tag: str) -> None:
        await asyncio.sleep(self.latency_ms / 1000.0)   # 无线时延
        if random.random() < self.loss_rate:
            logger.warning(f"wifi: packet loss on {tag} -> {dst_node}")
            return  # 丢包：调用方重试
        await super().push(data, dst_node, tag)

    async def measure_bandwidth(self, dst_node: str) -> float:
        # 无线带宽波动：在配置值附近抖动
        jitter = random.uniform(0.8, 1.2)
        return self._bandwidth * jitter

    async def estimate_transfer_time(self, data_bytes: int, dst_node: str) -> float:
        bw_bytes_per_ms = self._bandwidth * 1024 * 1024 / 1000.0
        return data_bytes / max(1.0, bw_bytes_per_ms) + self.latency_ms
