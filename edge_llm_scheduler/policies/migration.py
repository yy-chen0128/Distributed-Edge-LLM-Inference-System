"""迁移策略：节点离开时，其上的模型参数/KV 是否传走、传多少、传给谁。

已实现：
- PriorityMigration：KV 块优先级排序（PT×N = 重算时间×共享率），时间受限时只传部分

TODO（后续补充复杂逻辑）：
- token 级截断（极端时间紧时按 token 传已提交的 KV）
- 模型参数：从云端重载 vs 从其他设备传输的权衡
- 冗余复制策略（热点 KV 提前复制到备份节点）

默认实现 DefaultMigration：全传所有 KV 块，均匀分给剩余节点。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.storage import KVStore
from ..core.transport import Transport
from ..core.types import KVBlock, Node

logger = logging.getLogger(__name__)


@dataclass
class MigrationPlan:
    """迁移计划：哪些块去哪、丢哪些。"""
    block_moves: List[dict] = field(default_factory=list)  # [{block_hash, dst_location, priority}]
    block_drops: List[int] = field(default_factory=list)   # 丢弃的块哈希
    model_reload: bool = False                             # 模型参数是否需要重新装载


class MigrationPolicy(ABC):
    @abstractmethod
    async def decide(
        self,
        node_id: str,
        kv_block_ids: List[int],
        storage: KVStore,
        available_nodes: Optional[List[Node]] = None,
    ) -> MigrationPlan:
        """决定节点离开时 KV 块的去向。

        available_nodes: 剩余健康节点（目标候选）。None 时策略自己从 storage 索引推断。
        """

    @abstractmethod
    async def execute(
        self, plan: MigrationPlan, storage: KVStore, transport: Optional[Transport] = None
    ) -> None:
        """执行迁移计划。"""


class DefaultMigration(MigrationPolicy):
    """默认：全传，均匀分给剩余节点。"""

    async def decide(
        self,
        node_id: str,
        kv_block_ids: List[int],
        storage: KVStore,
        available_nodes: Optional[List[Node]] = None,
    ) -> MigrationPlan:
        plan = MigrationPlan()
        if not kv_block_ids:
            return plan
        index = await storage.get_index()
        # 收集所有可能的接收位置（除离开节点外）
        all_locs = set()
        for bid, locs in index.items():
            for l in locs:
                if not l.startswith(f"{node_id}:"):
                    all_locs.add(l)
        if not all_locs:
            all_locs.add("cloud:cpu")   # 无其他节点则回云端
        targets = sorted(all_locs)
        # 均匀分配
        for i, bid in enumerate(kv_block_ids):
            dst = targets[i % len(targets)]
            plan.block_moves.append({
                "block_hash": bid,
                "src_location": self._source_location(index, bid, node_id),
                "dst_location": dst,
                "priority": 0.0,
            })
        logger.info(f"migration plan for {node_id}: {len(plan.block_moves)} moves")
        return plan

    async def execute(
        self, plan: MigrationPlan, storage: KVStore, transport: Optional[Transport] = None
    ) -> None:
        for mv in plan.block_moves:
            bid = mv["block_hash"]
            src = mv.get("src_location")
            dst = mv["dst_location"]
            if not src:
                logger.warning(f"migration block {bid} has no source location")
                plan.block_drops.append(bid)
                continue
            try:
                await storage.move(bid, src, dst)
                logger.debug(f"migrated block {bid} -> {dst}")
            except Exception as e:
                logger.warning(f"migrate block {bid} failed: {e}; dropping")
                plan.block_drops.append(bid)
        for bid in plan.block_drops:
            try:
                await storage.evict(bid, location="*")
            except Exception:
                pass

    @staticmethod
    def _source_location(index: Dict[int, List[str]], block_hash: int, node_id: str) -> str:
        prefix = f"{node_id}:"
        for location in index.get(block_hash, []):
            if location.startswith(prefix):
                return location
        return f"{node_id}:gpu"


class PriorityMigration(MigrationPolicy):
    """PT×N 优先级迁移：按 KV 块价值降序传，时间受限时只传部分。

    价值公式（参考 Preble 驱逐价值 M_i = Σ PT_j × N_j）：
        priority(block) = reuse_count × prefill_time_ms
    - reuse_count: 历史共享请求数（热度）
    - prefill_time_ms: 重算这段 KV 的 prefill 耗时（越长越贵）

    时间预算：deadline_ms × bandwidth_mbps → 可传总字节。按价值降序，预算用完即丢。

    block_drop_hook: 可选回调，用于丢弃块时做记录（如补发 KV_MISS 事件）。
    """

    def __init__(
        self,
        deadline_ms: float = float("inf"),
        bandwidth_mbps: float = 100.0,
        target_check_fn: Optional[callable] = None,
    ) -> None:
        self.deadline_ms = deadline_ms
        self.bandwidth_mbps = bandwidth_mbps
        # 目标节点过滤回调：输入 Node，返回该节点能否接收一块 KV（显存够等）
        self.target_check_fn = target_check_fn or (lambda n: n.state.memory_free > 0)

    async def decide(
        self,
        node_id: str,
        kv_block_ids: List[int],
        storage: KVStore,
        available_nodes: Optional[List[Node]] = None,
    ) -> MigrationPlan:
        plan = MigrationPlan()
        if not kv_block_ids:
            return plan

        # ① 收集每块，算价值 PT×N
        index = await storage.get_index()
        valued: List[tuple] = []          # (block, priority)
        for bid in kv_block_ids:
            block = await self._load_block(storage, bid, node_id)
            if block is None:
                plan.block_drops.append(bid)
                continue
            priority = block.reuse_count * block.prefill_time_ms
            valued.append((block, priority))

        # ② 按价值降序
        valued.sort(key=lambda x: -x[1])

        # ③ 目标节点（剩余显存够 + 过滤回调）
        targets = self._select_targets(available_nodes)

        # ④ 时间受限：预算（毫秒）内按价值传，超预算丢
        budget_ms = self.deadline_ms
        for block, priority in valued:
            dst = self._assign_target(block, targets)
            if dst is None:
                plan.block_drops.append(block.block_hash)
                continue
            src = self._find_source_location(index, block.block_hash, node_id)
            cost_ms = await storage.estimate_move_cost(block, src, dst)
            if cost_ms <= budget_ms:
                plan.block_moves.append({
                    "block_hash": block.block_hash,
                    "src_location": src,
                    "dst_location": dst,
                    "priority": priority,
                    "cost_ms": cost_ms,
                })
                budget_ms -= cost_ms
            else:
                plan.block_drops.append(block.block_hash)

        logger.info(
            f"priority migration for {node_id}: {len(plan.block_moves)} moves, "
            f"{len(plan.block_drops)} drops (budget={self.deadline_ms}ms)"
        )
        return plan

    async def execute(
        self, plan: MigrationPlan, storage: KVStore, transport: Optional[Transport] = None
    ) -> None:
        for mv in plan.block_moves:
            bid = mv["block_hash"]
            src = mv.get("src_location")
            dst = mv["dst_location"]
            if not src:
                logger.warning(f"priority migration block {bid} has no source location")
                plan.block_drops.append(bid)
                continue
            try:
                await storage.move(bid, src, dst)
                logger.debug(f"priority-migrated block {bid} -> {dst}")
            except Exception as e:
                logger.warning(f"migrate block {bid} failed: {e}; dropping")
                plan.block_drops.append(bid)
        for bid in plan.block_drops:
            try:
                await storage.evict(bid, location="*")
            except Exception:
                pass

    # ---------- 内部 ----------

    async def _load_block(self, storage: KVStore, block_hash: int, node_id: str) -> Optional[KVBlock]:
        # 先试精确位置，再试任意位置
        for loc in (f"{node_id}:gpu", f"{node_id}:cpu", f"{node_id}:disk"):
            b = await storage.load(block_hash, loc)
            if b is not None:
                return b
        return None

    @staticmethod
    def _find_source_location(index: Dict[int, List[str]], block_hash: int, node_id: str) -> str:
        prefix = f"{node_id}:"
        for location in index.get(block_hash, []):
            if location.startswith(prefix):
                return location
        return f"{node_id}:gpu"

    def _select_targets(self, available_nodes: Optional[List[Node]]) -> List[Node]:
        if not available_nodes:
            return []
        return [n for n in available_nodes if n.state.alive and self.target_check_fn(n)]

    def _assign_target(self, block: KVBlock, targets: List[Node]) -> Optional[str]:
        """选目标：负载最轻的健康节点。返回 location 字符串或 None。"""
        if not targets:
            return None
        t = min(targets, key=lambda n: n.state.load)
        return f"{t.node_id}:gpu"
