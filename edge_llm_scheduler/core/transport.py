"""传输抽象接口（链路抽象）。

调度器把节点间通信视作"抽象链路"（Transport），不关心底层是有线还是无线。
适配在 backend 层：TCP（有线）/ WiFi（无线）/ mock。

所有方法返回前会等待传输完成；传输耗时由 backend 内部按带宽/时延模拟或实测。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Transport(ABC):
    """节点间数据链路抽象。"""

    @property
    @abstractmethod
    def bandwidth_mbps(self) -> float:
        """当前链路的带宽（MB/s），供成本估计使用。"""

    @abstractmethod
    async def push(self, data: bytes, dst_node: str, tag: str) -> None:
        """把数据推送到目标节点。tag 用于标识内容类型（如 'kv_block:123'）。"""

    @abstractmethod
    async def pull(self, src_node: str, tag: str) -> Optional[bytes]:
        """从源节点拉取数据。没有则返回 None。"""

    @abstractmethod
    async def measure_bandwidth(self, dst_node: str) -> float:
        """实测到目标节点的当前带宽（MB/s）。"""

    @abstractmethod
    async def estimate_transfer_time(self, data_bytes: int, dst_node: str) -> float:
        """估算传输 data_bytes 字节到 dst_node 的耗时（毫秒）。"""
