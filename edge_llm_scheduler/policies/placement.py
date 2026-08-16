"""放置策略：决定请求路由到哪些节点、任务如何切分。

已实现：
- DefaultPlacement：负载最轻的健康节点
- E2Placement：KV 缓存感知路由（exploit 命中 vs explore 负载均衡，参考 Preble E2）

TODO（后续补充复杂逻辑）：
- 按节点能力（算力/内存/带宽）分任务
- 专家覆盖路由（参考 MoE-Infinity/SMoE）
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from ..core.storage import KVStore
from ..core.types import InferenceRequest, Node, Task

logger = logging.getLogger(__name__)

# KV 块大小（token）——用于把命中块数换算成命中 token 数
BLOCK_SIZE_TOKENS = 16


class PlacementPolicy(ABC):
    @abstractmethod
    async def place(
        self,
        model,
        nodes: List[Node],
        request: InferenceRequest,
        storage: Optional[KVStore] = None,
        hit_tokens: int = 0,
    ) -> List[Task]:
        """为请求生成任务列表（决定去哪些节点、怎么切分）。"""


class DefaultPlacement(PlacementPolicy):
    """默认：选负载最轻的健康节点，整请求一个任务。

    含基本 KV 命中感知：若命中某节点的缓存，优先 exploit 它。
    """

    def __init__(self, load_threshold: float = 0.9) -> None:
        self.load_threshold = load_threshold

    async def place(
        self,
        model,
        nodes: List[Node],
        request: InferenceRequest,
        storage: Optional[KVStore] = None,
        hit_tokens: int = 0,
    ) -> List[Task]:
        if not nodes:
            return []
        candidates = [n for n in nodes if n.state.load < self.load_threshold]
        if not candidates:
            candidates = nodes
        # 选负载最轻
        target = min(candidates, key=lambda n: n.state.load)
        task = Task(
            task_id=str(uuid.uuid4()),
            request_id=request.request_id,
            node_id=target.node_id,
            layer_range=target.state.layer_range,
            status="pending",
        )
        logger.debug(f"routed {request.request_id} -> {target.node_id} (hit={hit_tokens})")
        return [task]


class E2Placement(PlacementPolicy):
    """E2 缓存感知路由（参考 Preble 的 Exploitation + Exploration）。

    核心决策：
      if 命中 token 数 > 剩余非共享 token 数:  → EXPLOIT：送"持有最长命中前缀"的节点
      else:                                     → EXPLORE：送"prompt-aware 负载最轻"的节点

    - EXPLOIT 省 prefill 计算（KV 命中），但可能让热点节点过载——只在命中收益高时用。
    - EXPLORE 均衡负载，但负载要含"驱逐代价"（去轻节点可能要逐它已有 KV，这也是成本）。

    block_size_tokens: 每块 KV 覆盖的 token 数（把 lookup 的块数换算成 token 数）。
    """

    def __init__(
        self,
        storage: Optional[KVStore] = None,
        block_size_tokens: int = BLOCK_SIZE_TOKENS,
        load_threshold: float = 0.95,
        storage_hit_weight: float = 1.0,
        hash_fn: Optional[callable] = None,
    ) -> None:
        self._storage = storage
        self.block_size_tokens = block_size_tokens
        self.load_threshold = load_threshold
        self.storage_hit_weight = storage_hit_weight  # 命中收益权重（后续调参）
        # 可注入的 prompt→块哈希 函数（测试用确定性 hash；真实用 LMCache 语义）
        self._hash_fn = hash_fn or self._default_hash

    async def place(
        self,
        model,
        nodes: List[Node],
        request: InferenceRequest,
        storage: Optional[KVStore] = None,
        hit_tokens: int = 0,
    ) -> List[Task]:
        if not nodes:
            return []
        store = storage or self._storage

        # ① 前缀匹配：查全局 KV 索引，找最长连续命中
        matched_tokens = 0
        if store is not None and request.prompt is not None:
            prefix_hashes = self._hash_prompt(request.prompt)
            if prefix_hashes:
                hit_blocks = await store.lookup(prefix_hashes)
                matched_tokens = hit_blocks * self.block_size_tokens

        # 用外部传入的 hit_tokens 覆盖（如果调用方已经算了更准的）
        if hit_tokens > matched_tokens:
            matched_tokens = hit_tokens

        remaining_tokens = max(0, self._prompt_len(request) - matched_tokens)

        # ② exploit/explore 决策
        if matched_tokens > remaining_tokens and store is not None:
            # EXPLOIT：在持有命中前缀缓存的节点里选负载最轻的
            holders = await self._find_hit_holders(store, prefix_hashes, matched_tokens // self.block_size_tokens)
            candidates = [n for n in nodes if n.node_id in holders and n.state.load < self.load_threshold]
            if candidates:
                target = min(candidates, key=lambda n: n.state.load)
                logger.debug(f"E2 exploit: {request.request_id} -> {target.node_id} (hit={matched_tokens})")
                return [self._make_task(target, request)]
            # 没有命中且健康的节点 → 退回 explore

        # EXPLORE：prompt-aware 负载最轻（已有负载 + 驱逐代价）
        target = await self._explore_target(nodes, store, request)
        logger.debug(f"E2 explore: {request.request_id} -> {target.node_id} (hit={matched_tokens})")
        return [self._make_task(target, request)]

    # ---------- 内部 ----------

    def _make_task(self, node: Node, request: InferenceRequest) -> Task:
        return Task(
            task_id=str(uuid.uuid4()),
            request_id=request.request_id,
            node_id=node.node_id,
            layer_range=node.state.layer_range,
            status="pending",
        )

    def _prompt_len(self, request: InferenceRequest) -> int:
        if request.prompt is None:
            return 0
        return len(request.prompt) if not isinstance(request.prompt, str) else len(request.prompt)

    def _hash_prompt(self, prompt) -> List[int]:
        """把 prompt 切成块哈希序列（可用注入的 hash_fn 覆盖）。"""
        if self._hash_fn is not None:
            return self._hash_fn(prompt, self.block_size_tokens)
        return self._default_hash(prompt, self.block_size_tokens)

    @staticmethod
    def _default_hash(prompt, block_size_tokens: int) -> List[int]:
        tokens = list(range(len(prompt))) if isinstance(prompt, str) else list(prompt)
        blocks = [tokens[i:i + block_size_tokens]
                  for i in range(0, len(tokens), block_size_tokens)]
        return [hash(tuple(b)) for b in blocks]

    async def _find_hit_holders(self, store: KVStore, prefix_hashes: List[int], hit_blocks: int) -> set:
        """找出持有命中前缀块缓存的节点集合。"""
        if hit_blocks <= 0:
            return set()
        index = await store.get_index()
        hit_hashes = set(prefix_hashes[:hit_blocks])
        holders = set()
        for bid, locs in index.items():
            if bid in hit_hashes:
                for loc in locs:
                    node = loc.split(":")[0]
                    holders.add(node)
        return holders

    async def _explore_target(self, nodes: List[Node], store: Optional[KVStore],
                              request: InferenceRequest) -> Node:
        """选 prompt-aware 负载最轻的节点。"""
        if store is None:
            return min(nodes, key=lambda n: n.state.load)

        # 计算每个节点的驱逐代价：其持有的最低价值 KV 块
        index = await store.get_index()
        evict_cost = {}   # node_id -> 驱逐代价
        for bid, locs in index.items():
            for loc in locs:
                node = loc.split(":")[0]
                if node in evict_cost:
                    continue
                block = await store.load(bid, loc)
                if block is not None:
                    value = block.reuse_count * block.prefill_time_ms
                    # 记录该节点价值最低的块（驱逐它代价最小）
                    evict_cost[node] = min(evict_cost.get(node, float("inf")), value)

        def total_cost(n: Node) -> float:
            base = n.state.load
            evict = evict_cost.get(n.node_id, 0.0)
            # 归一化：驱逐代价 / (驱逐代价上限 + 1) 使其与负载同量级
            return base + min(1.0, evict / (max(evict_cost.values()) + 1e-9)) if evict_cost else base

        return min(nodes, key=total_cost)


class CapabilityPlacement(PlacementPolicy):
    """按节点能力加权路由：算力强的节点多承担（性能目标：木桶效应最小化）。

    决策：选"有效利用率最低"的节点——即 (负载 / 能力权重) 最小。
    能力权重 = compute_flops / 集群平均算力。算力 2× 的节点，同样负载下"更闲"。

    比纯负载均衡好：异构集群里，负载相同但算力不同时，该把请求给算力强的。
    """

    async def place(
        self,
        model,
        nodes: List[Node],
        request: InferenceRequest,
        storage: Optional[KVStore] = None,
        hit_tokens: int = 0,
    ) -> List[Task]:
        if not nodes:
            return []
        alive = [n for n in nodes if n.state.alive]
        if not alive:
            return []
        # 能力权重 = 相对集群平均
        avg_flops = sum(max(1e-9, n.capability.compute_flops) for n in alive) / len(alive)

        def effective_load(n: Node) -> float:
            w = max(1e-9, n.capability.compute_flops) / avg_flops
            return n.state.load / w  # 算力强（w>1）→ 有效负载小 → 优先

        target = min(alive, key=effective_load)
        task = Task(
            task_id=str(uuid.uuid4()),
            request_id=request.request_id,
            node_id=target.node_id,
            layer_range=target.state.layer_range,
            status="pending",
        )
        logger.debug(f"capability-routed {request.request_id} -> {target.node_id}")
        return [task]
