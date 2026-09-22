"""低耦合的边缘分层执行器协议。

该协议刻意不依赖 vLLM 的 EngineCore、SchedulerOutput 或 Worker 内部对象。控制面
只要求阶段运行时能够准备/激活某个层区间，并执行携带 activation 的 Task。真实实现
可以是静态 vLLM PP 组的 sidecar、Transformers worker 或专用 CUDA runtime。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .types import GenerationResult, StageAssignment, Task


@dataclass(frozen=True)
class StagePrepareResult:
    node_id: str
    pipeline_epoch: int
    layer_range: tuple[int, int]
    model_source: str


class StageRuntime(ABC):
    """节点侧阶段执行器的最小契约。"""

    @abstractmethod
    async def prepare_epoch(
        self,
        assignment: StageAssignment,
        model_source: str = "model-store",
    ) -> StagePrepareResult:
        """预装指定层区间，但不立刻把它用于新请求。"""

    @abstractmethod
    async def activate_epoch(self, pipeline_epoch: int) -> None:
        """将已准备的阶段标记为当前可接收的新请求。"""

    @abstractmethod
    async def retire_epoch(self, pipeline_epoch: int) -> None:
        """在旧请求排空后释放一个旧阶段（也可为该节点最后的 active epoch）。"""

    @abstractmethod
    async def generate(self, task: Task) -> GenerationResult:
        """执行一个阶段任务。"""

    @property
    @abstractmethod
    def node_id(self) -> str:
        """运行时所属节点。"""
