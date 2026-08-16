"""重并行化策略：节点加入/离开后，流水线拓扑与并行度如何调整。

已实现：
- CapabilityReparallelization：按节点算力比例重新切分层区间（轻量，无需 profiling）

性能目标：让每个节点的计算负载均衡——算力强的节点分更多层，避免木桶效应。
这是"最终系统性能良好"的基础：节点变化后若不重排，强节点空闲、弱节点过载。

TODO（后续补充）：
- 完整 (D,P,M,B) 枚举 + 延迟模型（参考 SpotServe StrategyOptimizer）
- 流水线拓扑变化（pipeline 拼接/断开，参考 SpotServe HotSwitch）
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..core.types import ModelPlacement, ModelSpec, Node

logger = logging.getLogger(__name__)


class ReparallelizationPolicy(ABC):
    @abstractmethod
    async def reconfigure(
        self,
        model: Optional[ModelSpec],
        nodes: List[Node],
        current_placements: Dict[str, ModelPlacement],
    ) -> List[ModelPlacement]:
        """根据当前节点集合重新计算放置计划。"""


class DefaultReparallelization(ReparallelizationPolicy):
    """默认：保持现有放置不变。"""

    async def reconfigure(
        self,
        model: Optional[ModelSpec],
        nodes: List[Node],
        current_placements: Dict[str, ModelPlacement],
    ) -> List[ModelPlacement]:
        return list(current_placements.values())


class CapabilityReparallelization(ReparallelizationPolicy):
    """按节点算力比例重新切分层区间。

    原理：把模型的总层数按各节点的算力占比分配——算力强的多分。
    结果：每节点预计执行时间相近（负载均衡），整体吞吐最优。

    需要 model.num_layers。若 model 为 None 或节点算力不可用则退回默认。
    """

    async def reconfigure(
        self,
        model: Optional[ModelSpec],
        nodes: List[Node],
        current_placements: Dict[str, ModelPlacement],
    ) -> List[ModelPlacement]:
        if model is None or not nodes:
            return list(current_placements.values())

        alive = [n for n in nodes if n.state.alive]
        if not alive:
            return []

        # 计算每节点算力权重
        total_flops = sum(max(1.0, n.capability.compute_flops) for n in alive)
        weights = {n.node_id: max(1.0, n.capability.compute_flops) / total_flops for n in alive}

        # 按权重分配层数（整数化，把余数给权重最大的）
        num_layers = model.num_layers
        raw = {nid: w * num_layers for nid, w in weights.items()}
        assigned = {nid: int(r) for nid, r in raw.items()}
        remaining = num_layers - sum(assigned.values())
        if remaining > 0:
            # 按小数部分从大到小补层
            for nid, _ in sorted(raw.items(), key=lambda kv: -((kv[1] - assigned[kv[0]])))[:remaining]:
                assigned[nid] += 1

        # 生成层区间（按节点顺序累积）
        placements = []
        start = 0
        for n in alive:
            cnt = assigned.get(n.node_id, 0)
            if cnt <= 0:
                continue
            placements.append(ModelPlacement(
                node_id=n.node_id,
                layer_range=(start, start + cnt),
            ))
            start += cnt
        # 若总分配少于模型层数，把剩余给最后一个
        if start < num_layers and placements:
            placements[-1].layer_range = (
                placements[-1].layer_range[0],
                num_layers,
            )

        logger.info(
            f"reparallelized {num_layers} layers over {len(placements)} nodes: "
            + ", ".join(f"{p.node_id}={p.layer_range}" for p in placements)
        )
        return placements
