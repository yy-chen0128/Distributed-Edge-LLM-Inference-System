"""Mock KV 存储：dict 后端，模拟 KVStore 接口。

数据存内存 dict：block_hash -> {location -> KVBlock}。
lookup 模拟前缀命中：按连续哈希匹配。
用于无硬件验证框架的 KV 管理逻辑。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from ..core.storage import KVStore
from ..core.types import KVBlock

logger = logging.getLogger(__name__)


class MockKVStore(KVStore):
    def __init__(self) -> None:
        # block_hash -> {location: KVBlock}
        self._store: Dict[int, Dict[str, KVBlock]] = {}
        # 前缀顺序记录（用于 lookup 连续匹配）
        self._prefix_order: List[int] = []
        # 默认带宽（用于估算迁移耗时）
        self.bandwidth_mbps: float = 100.0

    async def save(self, block: KVBlock, location: str) -> None:
        self._store.setdefault(block.block_hash, {})[location] = block
        if block.block_hash not in self._prefix_order:
            self._prefix_order.append(block.block_hash)

    async def load(self, block_hash: int, location: str) -> Optional[KVBlock]:
        return self._store.get(block_hash, {}).get(location)

    async def lookup(self, prefix_hashes: List[int], locations: Optional[List[str]] = None) -> int:
        """连续命中块数：从头部开始，逐块检查是否存在于指定位置。

        请求自己的哈希序列即"前缀"，只要某块存在（在指定位置内）就命中，
        从头连续命中多少块就返回多少。
        """
        hit = 0
        for h in prefix_hashes:
            if self._has_block(h, locations):
                hit += 1
            else:
                break
        return hit

    async def move(self, block_hash: int, src: str, dst: str) -> None:
        """搬移：从 src 移到 dst（搬移语义，删除源）。"""
        locs = self._store.get(block_hash)
        if locs is None:
            raise KeyError(f"block {block_hash} not found")
        block = locs.get(src)
        if block is None:
            # 容忍源位置不精确：从任意位置取
            block = self._first_block(block_hash)
        if block is None:
            raise KeyError(f"block {block_hash} has no data")
        locs[dst] = block
        # 搬移语义：删除除 dst 外的所有位置（不管 src 精不精确）
        for k in [k for k in locs if k != dst]:
            del locs[k]

    async def evict(self, block_hash: int, location: str) -> None:
        if location == "*":
            self._store.pop(block_hash, None)
            if block_hash in self._prefix_order:
                self._prefix_order.remove(block_hash)
            return
        self._store.get(block_hash, {}).pop(location, None)
        if not self._store.get(block_hash):
            self._store.pop(block_hash, None)
            if block_hash in self._prefix_order:
                self._prefix_order.remove(block_hash)

    async def get_index(self) -> Dict[int, List[str]]:
        return {bid: list(locs.keys()) for bid, locs in self._store.items()}

    async def estimate_move_cost(self, block: KVBlock, src: str, dst: str) -> float:
        return block.byte_size / max(1.0, self.bandwidth_mbps) * 0.001  # ms

    def _has_block(self, block_hash: int, locations: Optional[List[str]]) -> bool:
        if block_hash not in self._store:
            return False
        if locations is None:
            return True
        return any(l in self._store[block_hash] for l in locations)

    def _first_block(self, block_hash: int) -> Optional[KVBlock]:
        locs = self._store.get(block_hash)
        if not locs:
            return None
        return next(iter(locs.values()))
