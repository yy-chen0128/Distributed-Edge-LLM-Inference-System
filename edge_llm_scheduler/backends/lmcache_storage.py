"""LMCache 存储后端：把 KVStore 抽象映射到 LMCacheEngine。

对接点（探索结论）：
- 高层 API：`LMCacheEngine`（lmcache/v1/cache_engine.py）
  - store(tokens, ...) / retrieve(tokens, ...) / lookup(tokens, ...) / move(...) / clear(...)
- 对接单元：`CacheEngineKey`（token 哈希键）+ `MemoryObj`（KV 数据）
- 位置字符串：我们把 "node:id:gpu" / "node:id:cpu" / "node:id:disk" 映射为 LMCache 的
  GPU(显存)/CPU(host)/盘 存储层。

注意：LMCache 的键是基于 token 序列哈希的，跨进程必须统一 PYTHONHASHSEED。
真实使用需要本机跑 LMCache 实例；本文件提供对接骨架，mock 环境用 MockKVStore。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..core.storage import KVStore
from ..core.types import KVBlock

logger = logging.getLogger(__name__)


class LMCacheStore(KVStore):
    """把 KVStore 接口适配到 LMCacheEngine。

    lm_engine: LMCacheEngine 实例（由 LMCacheManager 创建，见 lmcache/v1/manager.py）。
    """

    def __init__(self, lm_engine=None, model_name: str = "default") -> None:
        self._lm = lm_engine
        self.model_name = model_name
        self._bandwidth_mbps = 100.0
        # LMCacheEngine 本身通常不提供跨节点全局索引；该目录属于调度器，
        # 记录 block -> location，供 E2、迁移和故障恢复使用。
        self._catalog: Dict[int, Dict[str, KVBlock]] = {}

    # ---------- KVStore 实现 ----------

    async def save(self, block: KVBlock, location: str) -> None:
        if self._lm is None:
            raise RuntimeError("LMCacheStore not attached to LMCacheEngine")
        # location 形如 "node_a:gpu" —— 决定存入哪个存储层
        tier = location.split(":")[-1] if ":" in location else "gpu"
        tokens = list(block.tokens) or [block.block_hash] * block.num_tokens
        # LMCacheEngine.store 是同步方法；这里包一层（真实环境注意线程安全）
        await self._run_in_loop(lambda: self._lm.store(tokens=tokens, request_configs=None))
        self._catalog.setdefault(block.block_hash, {})[location] = block
        logger.debug(f"lmcache save block {block.block_hash} tier={tier}")

    async def load(self, block_hash: int, location: str) -> Optional[KVBlock]:
        if self._lm is None:
            raise RuntimeError("LMCacheStore not attached")
        known = next(iter(self._catalog.get(block_hash, {}).values()), None)
        tokens = list(known.tokens) if known and known.tokens else [block_hash]
        mask = await self._run_in_loop(lambda: self._lm.retrieve(tokens=tokens))
        # retrieve 返回 bool 掩码：True 表示命中
        if mask is not None and len(mask) > 0 and bool(mask[0]):
            return known or KVBlock(block_hash=block_hash, num_tokens=len(tokens), byte_size=0, tokens=tokens)
        return None

    async def lookup(self, prefix_hashes: List[int], locations: Optional[List[str]] = None) -> int:
        if self._lm is None:
            raise RuntimeError("LMCacheStore not attached")
        if locations is not None:
            hit = 0
            for block_hash in prefix_hashes:
                if not any(
                    block_hash in self._catalog and location in self._catalog[block_hash]
                    for location in locations
                ):
                    break
                hit += 1
            return hit
        hit = await self._run_in_loop(lambda: self._lm.lookup(tokens=prefix_hashes))
        return int(hit or 0)

    async def move(self, block_hash: int, src: str, dst: str) -> None:
        if self._lm is None:
            raise RuntimeError("LMCacheStore not attached")
        # LMCacheEngine.move(tokens, old_position, new_position, event_id)
        known = self._catalog.get(block_hash, {})
        if known and src not in known:
            raise KeyError(f"block {block_hash} not found at source {src}")
        block = known.get(src)
        tokens = list(block.tokens) if block and block.tokens else [block_hash]
        await self._run_in_loop(
            lambda: self._lm.move(tokens=tokens, old_position=src, new_position=(dst, ""), event_id="scheduler")
        )
        if known:
            known[dst] = known.pop(src)

    async def evict(self, block_hash: int, location: str) -> None:
        if self._lm is None:
            raise RuntimeError("LMCacheStore not attached")
        known = self._catalog.get(block_hash, {})
        block = next(iter(known.values()), None)
        tokens = list(block.tokens) if block and block.tokens else [block_hash]
        locations = None if location == "*" else [location]
        await self._run_in_loop(lambda: self._lm.clear(tokens=tokens, locations=locations))
        if location == "*":
            self._catalog.pop(block_hash, None)
        else:
            known.pop(location, None)
            if not known:
                self._catalog.pop(block_hash, None)

    async def get_index(self) -> Dict[int, List[str]]:
        return {bid: list(locations.keys()) for bid, locations in self._catalog.items()}

    async def estimate_move_cost(self, block: KVBlock, src: str, dst: str) -> float:
        return block.byte_size / max(1.0, self._bandwidth_mbps) / 1000.0  # 秒→ms

    # ---------- 辅助 ----------

    async def _run_in_loop(self, fn):
        """把同步的 LMCacheEngine 调用放到事件循环线程执行。"""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)
