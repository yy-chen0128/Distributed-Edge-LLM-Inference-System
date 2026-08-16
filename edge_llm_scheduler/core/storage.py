"""KV 存储抽象接口。

框架只依赖本接口（KVStore），不直接依赖 LMCache 的 StorageBackendInterface
（它耦合 torch/MemoryObj）。LMCache 作为 KVStore 的一个 backend 实现。
这样框架可测、可换后端。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .types import KVBlock


class KVStore(ABC):
    """KV 缓存的统一抽象：存、取、查、迁、逐。

    location 是存储位置字符串，如 "node_a:gpu" / "node_a:cpu" / "node_a:disk"。
    框架层把位置当作不透明标识，底层 backend 解释它。
    """

    @abstractmethod
    async def save(self, block: KVBlock, location: str) -> None:
        """把一个 KV 块存入指定位置。"""

    @abstractmethod
    async def load(self, block_hash: int, location: str) -> Optional[KVBlock]:
        """从指定位置取回一个 KV 块。"""

    @abstractmethod
    async def lookup(self, prefix_hashes: List[int], locations: Optional[List[str]] = None) -> int:
        """前缀命中查询：给定前缀块哈希序列，返回连续命中的块数。

        locations 非空时只在指定位置里查（如只查某节点的 GPU 层）。
        """

    @abstractmethod
    async def move(self, block_hash: int, src: str, dst: str) -> None:
        """跨位置搬移一个 KV 块（节点离开/冗余复制时用）。"""

    @abstractmethod
    async def evict(self, block_hash: int, location: str) -> None:
        """从某位置逐出一个 KV 块。"""

    @abstractmethod
    async def get_index(self) -> Dict[int, List[str]]:
        """返回全局索引：block_hash -> 持有它的位置列表。

        这是调度器做"缓存感知路由"的数据来源。
        """

    @abstractmethod
    async def estimate_move_cost(self, block: KVBlock, src: str, dst: str) -> float:
        """估算搬移一块 KV 的耗时（毫秒）——路由/迁移决策用。"""
