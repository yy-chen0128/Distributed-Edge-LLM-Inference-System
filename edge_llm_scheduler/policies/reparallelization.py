"""重并行化策略：节点加入/离开后，流水线拓扑与并行度如何调整。

TODO（后续补充复杂逻辑）：
- 节点变化后重新计算 (D, P, M, B) 并行度配置（参考 SpotServe StrategyOptimizer）
- 流水线拓扑变化（pipeline 拼接/断开，参考 SpotServe HotSwitch）
- 层区间重划分（新节点加入时把层拆给它，参考 Helix placement）

默认实现 DefaultReparallelization：不改变现有放置（先跑通框架）。
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
    """默认：保持现有放置不变。

    TODO：替换为按节点能力重新划分层区间 + 并行度调整。
    """

    async def reconfigure(
        self,
        model: Optional[ModelSpec],
        nodes: List[Node],
        current_placements: Dict[str, ModelPlacement],
    ) -> List[ModelPlacement]:
        return list(current_placements.values())
