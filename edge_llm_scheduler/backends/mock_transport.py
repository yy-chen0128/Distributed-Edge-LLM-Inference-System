"""Mock 传输：模拟链路（带宽/时延），无真实网络。

用带宽和时延模拟 push/pull 的耗时，验证调度器对传输成本估计的逻辑。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from ..core.transport import Transport

logger = logging.getLogger(__name__)


class MockTransport(Transport):
    def __init__(
        self,
        bandwidth_mbps: float = 100.0,
        latency_ms: float = 10.0,
        loss_rate: float = 0.0,
    ) -> None:
        self._bandwidth = bandwidth_mbps
        self._latency_ms = latency_ms
        self._loss_rate = loss_rate
        # tag -> bytes（模拟对端数据）
        self._inbox: Dict[str, bytes] = {}

    @property
    def bandwidth_mbps(self) -> float:
        return self._bandwidth

    async def push(self, data: bytes, dst_node: str, tag: str) -> None:
        # 模拟传输耗时 = 数据量 / 带宽 + 时延
        delay = self._transfer_delay(len(data))
        await asyncio.sleep(delay / 1000.0)
        if self._loss_rate > 0 and _rand() < self._loss_rate:
            logger.warning(f"mock transport: packet loss on {tag}")
            return
        self._inbox[f"{dst_node}:{tag}"] = data

    async def pull(self, src_node: str, tag: str) -> Optional[bytes]:
        key = f"{src_node}:{tag}"
        data = self._inbox.pop(key, None)
        return data

    async def measure_bandwidth(self, dst_node: str) -> float:
        # 模拟一次探测的时延
        await asyncio.sleep(self._latency_ms / 1000.0)
        return self._bandwidth

    async def estimate_transfer_time(self, data_bytes: int, dst_node: str) -> float:
        return self._transfer_delay(data_bytes)

    def _transfer_delay(self, data_bytes: int) -> float:
        # 带宽单位 MB/s（M = 2^20）。耗时(ms) = 字节 / (MB/s * 2^20 字节/s) * 1000
        bw_bytes_per_ms = self._bandwidth * 1024 * 1024 / 1000.0
        return data_bytes / max(1.0, bw_bytes_per_ms) + self._latency_ms


def _rand() -> float:
    import random
    return random.random()
