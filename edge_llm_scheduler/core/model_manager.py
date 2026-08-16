"""模型装载协调。

用户明确要求的能力：
- 控制装哪些层（layer_range）
- 控制装哪些专家（expert_ids）
- 控制专家装入哪级存储（GPU / CPU / 盘）

机制：本模块维护"每节点装了哪些层/哪些专家、在哪级存储"，并提供装载/卸载接口。
真实的参数传输走 Transport（抽象链路），存储层级用 location 字符串表示。
装什么、装哪、装多少 由上层策略（PlacementPolicy）决定——本模块只执行。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from .node_manager import NodeManager
from .transport import Transport
from .types import ModelPlacement, ModelSpec, Node

logger = logging.getLogger(__name__)


class ModelManager:
    """模型装载状态与操作。

    placement 记录：node_id -> 该节点已装载的 ModelPlacement（层区间 + 专家分级）。
    """

    def __init__(
        self,
        model: ModelSpec,
        node_manager: NodeManager,
        transport: Optional[Transport] = None,
    ) -> None:
        self.model = model
        self._node_mgr = node_manager
        self._transport = transport
        # node_id -> ModelPlacement（已装载的）
        self._placements: Dict[str, ModelPlacement] = {}
        self._loading: set = set()          # 装载中的 node_id（防并发重复装载）

    # ---------- 查询 ----------

    def get_placement(self, node_id: str) -> Optional[ModelPlacement]:
        return self._placements.get(node_id)

    def all_placements(self) -> Dict[str, ModelPlacement]:
        return dict(self._placements)

    def node_has_layer(self, node_id: str, layer_id: int) -> bool:
        p = self._placements.get(node_id)
        if not p or not p.layer_range:
            return False
        return p.layer_range[0] <= layer_id < p.layer_range[1]

    def node_has_expert(self, node_id: str, expert_id: int, tier: str = "gpu") -> bool:
        p = self._placements.get(node_id)
        if not p:
            return False
        if tier == "gpu":
            return expert_id in p.experts_gpu
        elif tier == "cpu":
            return expert_id in p.experts_cpu
        return False

    def covered_layer_ranges(self) -> List[tuple]:
        """当前全部已装载的层区间（用于检查模型是否覆盖完整）。"""
        return [p.layer_range for p in self._placements.values() if p.layer_range]

    # ---------- 装载 / 卸载 ----------

    async def load_model(self, placements: List[ModelPlacement]) -> None:
        """按放置计划批量装载。"""
        await asyncio.gather(*(self.load_placement(p) for p in placements))

    async def load_placement(self, placement: ModelPlacement) -> None:
        """装载一个节点的放置计划（层 + 专家 + 分级）。"""
        node_id = placement.node_id
        if node_id in self._loading:
            return
        self._loading.add(node_id)
        try:
            node = self._node_mgr.get(node_id)
            if node is None:
                raise ValueError(f"unknown node: {node_id}")
            await self._check_memory(node, placement)
            # 逐部分装载：层 → GPU 专家 → CPU 专家
            if placement.layer_range:
                await self._load_layers(node_id, placement.layer_range)
            for eid in placement.experts_gpu:
                await self._load_expert(node_id, eid, tier="gpu")
            for eid in placement.experts_cpu:
                await self._load_expert(node_id, eid, tier="cpu")
            self._placements[node_id] = placement
            logger.info(
                f"loaded {node_id}: layers={placement.layer_range} "
                f"gpu_experts={len(placement.experts_gpu)} cpu_experts={len(placement.experts_cpu)}"
            )
        finally:
            self._loading.discard(node_id)

    async def load_layers(self, node_id: str, layer_range: Tuple[int, int], source: str = "cloud") -> None:
        """装载指定层区间。source 表示参数来源（'cloud'/'peer'）。"""
        node = self._require_node(node_id)
        node.state.layer_range = layer_range
        # 若配置了传输层且源是远端，这里会走抽象链路（真实实现时）
        logger.info(f"load layers {layer_range} to {node_id} (source={source})")

    async def load_experts(self, node_id: str, expert_ids: List[int], tier: str = "gpu") -> None:
        """装载指定专家到指定存储层级。tier in {'gpu','cpu','disk'}。"""
        node = self._require_node(node_id)
        if tier == "gpu":
            node.state.expert_ids.update(expert_ids)
        elif tier == "cpu":
            node.state.expert_ids_cpu.update(expert_ids)
        logger.info(f"load {len(expert_ids)} experts to {node_id} tier={tier}")

    async def unload(self, node_id: str, layer_range: Optional[tuple] = None, expert_ids: Optional[List[int]] = None) -> None:
        """卸载指定层/专家（节点离开/重放置时释放）。"""
        node = self._node_mgr.get(node_id)
        if node is None:
            return
        if layer_range is not None:
            node.state.layer_range = None
            node.state.kv_blocks = 0
        if expert_ids is not None:
            node.state.expert_ids.difference_update(expert_ids)
        p = self._placements.get(node_id)
        if p:
            if layer_range is not None:
                p.layer_range = None
            if expert_ids is not None:
                p.experts_gpu = [e for e in p.experts_gpu if e not in expert_ids]
        logger.info(f"unloaded from {node_id}: layers={layer_range} experts={expert_ids}")

    # ---------- 内部 ----------

    def _require_node(self, node_id: str) -> Node:
        node = self._node_mgr.get(node_id)
        if node is None:
            raise ValueError(f"unknown node: {node_id}")
        return node

    async def _load_layers(self, node_id: str, layer_range: Tuple[int, int]) -> None:
        # 每层大小 = attention + expert 平均权重；简化：按总参数均分
        layer_bytes = self.model.total_params_bytes // max(1, self.model.num_layers)
        bytes_needed = (layer_range[1] - layer_range[0]) * layer_bytes
        await self._consume_memory(node_id, bytes_needed)

    async def _load_expert(self, node_id: str, expert_id: int, tier: str) -> None:
        expert = next((e for e in self.model.experts if e.expert_id == expert_id), None)
        size = expert.size_bytes if expert else 0
        if tier == "gpu":
            await self._consume_memory(node_id, size)
        # CPU/disk 不占 GPU 显存

    async def _consume_memory(self, node_id: str, bytes_needed: int) -> None:
        node = self._require_node(node_id)
        if bytes_needed > node.state.memory_free:
            raise MemoryError(
                f"node {node_id} insufficient memory: need {bytes_needed}, free {node.state.memory_free}"
            )
        node.state.memory_free -= bytes_needed

    async def _check_memory(self, node: Node, placement: ModelPlacement) -> None:
        """装载前预检显存是否足够（粗略检查，精确计算后续补充）。"""
        # 注意力权重可全复制（小），不计入；层/专家占 GPU
        est = 0
        if placement.layer_range:
            layer_bytes = self.model.total_params_bytes // max(1, self.model.num_layers)
            est += (placement.layer_range[1] - placement.layer_range[0]) * layer_bytes
        for eid in placement.experts_gpu:
            expert = next((e for e in self.model.experts if e.expert_id == eid), None)
            est += expert.size_bytes if expert else 0
        if est > node.capability.memory_total:
            raise MemoryError(
                f"placement exceeds node {node.node_id} capacity: need {est}, have {node.capability.memory_total}"
            )
