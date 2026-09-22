"""边缘分层流水线的 epoch 化重配置协调器。

该模块是 vLLM 之外的控制面：它只编排节点侧 StageRuntime、模型层区间账本和
KVStore。这样节点加入/离开、异构重分层和蓝绿切流不依赖 vLLM 的私有 Worker
对象或固定的 torch.distributed process group。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .model_manager import ModelManager
from .node_manager import NodeManager
from .stage_runtime import StageRuntime
from .storage import KVStore
from .types import ModelPlacement, PipelinePlan, StageAssignment
from ..policies.placement import LayeredPipelinePlacement
from ..policies.reparallelization import ReparallelizationPolicy


@dataclass(frozen=True)
class StateTransferRecord:
    """一次重配置中保留下来的 KV 块副本。"""

    block_hash: int
    layer_range: tuple[int, int]
    source: str
    destination: str


@dataclass
class ReconfigurationReport:
    pipeline_epoch: int
    placements: List[ModelPlacement]
    prepared_nodes: List[str] = field(default_factory=list)
    kv_copies: List[StateTransferRecord] = field(default_factory=list)


class PipelineReconfigurationCoordinator:
    """以“预装 -> 复制状态 -> 切流”的顺序应用一个分层计划。"""

    def __init__(
        self,
        node_manager: NodeManager,
        model_manager: ModelManager,
        storage: KVStore,
        placement_policy: LayeredPipelinePlacement,
        runtimes: Optional[Dict[str, StageRuntime]] = None,
    ) -> None:
        self.node_manager = node_manager
        self.model_manager = model_manager
        self.storage = storage
        self.placement_policy = placement_policy
        self.runtimes: Dict[str, StageRuntime] = dict(runtimes or {})
        self.current_plan: Optional[PipelinePlan] = None
        self._next_epoch = max(0, placement_policy.pipeline_epoch)

    def attach_runtime(self, runtime: StageRuntime) -> None:
        self.runtimes[runtime.node_id] = runtime

    async def apply_plan(
        self,
        placements: List[ModelPlacement],
        model_source: str = "model-store",
    ) -> ReconfigurationReport:
        """创建下一 epoch 并在所有新 stage 就绪后再让入口切流。"""
        self._validate_plan(placements)
        next_epoch = self._next_epoch + 1
        source_by_node = self._source_nodes_for(placements)

        prepared: List[str] = []
        for placement in placements:
            runtime = self.runtimes.get(placement.node_id)
            if runtime is None:
                raise ValueError(f"no StageRuntime registered for {placement.node_id}")
            assignment = StageAssignment(
                pipeline_epoch=next_epoch,
                node_id=placement.node_id,
                layer_range=placement.layer_range,
                source_node=source_by_node.get(placement.node_id),
            )
            source = assignment.source_node or model_source
            await runtime.prepare_epoch(assignment, model_source=source)
            prepared.append(placement.node_id)

        # 节点运行时完成预装后，才更新调度侧账本；随后复制旧层对应的 KV。
        await self.model_manager.apply_pipeline_plan(placements)
        kv_copies = await self._copy_reusable_kv(placements)

        for placement in placements:
            await self.runtimes[placement.node_id].activate_epoch(next_epoch)

        plan = PipelinePlan(pipeline_epoch=next_epoch, placements=list(placements))
        self.current_plan = plan
        self._next_epoch = next_epoch
        self.placement_policy.set_plan(plan)

        # 不在此处释放 old_plan：包括已不参与新计划的节点，仍可能处理绑定旧
        # epoch 的在途请求。入口完成 drain 后由 drain_old_epoch 显式退役。

        return ReconfigurationReport(
            pipeline_epoch=next_epoch,
            placements=list(placements),
            prepared_nodes=prepared,
            kv_copies=kv_copies,
        )

    async def drain_old_epoch(self, pipeline_epoch: int) -> None:
        """由入口确认旧 epoch 已无在途请求后调用，释放其节点侧资源。"""
        if self.current_plan is not None and pipeline_epoch == self.current_plan.pipeline_epoch:
            raise ValueError("cannot drain the active pipeline epoch")
        for runtime in self.runtimes.values():
            await runtime.retire_epoch(pipeline_epoch)

    async def reconfigure_for_topology(
        self,
        policy: Optional[ReparallelizationPolicy] = None,
        model_source: str = "model-store",
    ) -> ReconfigurationReport:
        """使用给定策略，根据当前存活节点重新计算并应用层区间。"""
        if policy is None:
            from ..policies.reparallelization import CapabilityReparallelization

            policy = CapabilityReparallelization()
        placements = await policy.reconfigure(
            self.model_manager.model,
            self.node_manager.alive_nodes(),
            self.model_manager.all_placements(),
        )
        return await self.apply_plan(placements, model_source=model_source)

    def _validate_plan(self, placements: List[ModelPlacement]) -> None:
        self.model_manager.validate_pipeline_coverage(placements)
        for placement in placements:
            node = self.node_manager.get(placement.node_id)
            if node is None or not node.state.alive:
                raise ValueError(f"pipeline plan includes unavailable node: {placement.node_id}")

    def _source_nodes_for(self, placements: List[ModelPlacement]) -> Dict[str, str]:
        if self.current_plan is None:
            return {}
        old_by_range = {
            placement.layer_range: placement.node_id
            for placement in self.current_plan.placements
            if placement.layer_range is not None
        }
        return {
            placement.node_id: old_by_range[placement.layer_range]
            for placement in placements
            if placement.layer_range in old_by_range
            and old_by_range[placement.layer_range] != placement.node_id
        }

    async def _copy_reusable_kv(
        self,
        placements: List[ModelPlacement],
    ) -> List[StateTransferRecord]:
        """按 KV 自身的 layer_range 复制到新拥有该区间的节点。

        这里采用 copy 而非 move：旧 epoch 仍可能有在途请求，待 drain 后再回收。
        对 LMCache 后端，这个 catalog 动作可映射为 connector/缓存服务的异步副本。
        """
        copies: List[StateTransferRecord] = []
        index = await self.storage.get_index()
        for block_hash, locations in index.items():
            for source in locations:
                block = await self.storage.load(block_hash, source)
                if block is None or block.layer_range is None:
                    continue
                owner = self._owner_for_range(block.layer_range, placements)
                if owner is None:
                    continue
                destination = f"{owner.node_id}:gpu"
                if source == destination:
                    continue
                await self.storage.save(block, destination)
                copies.append(
                    StateTransferRecord(
                        block_hash=block_hash,
                        layer_range=block.layer_range,
                        source=source,
                        destination=destination,
                    )
                )
                break
        return copies

    @staticmethod
    def _owner_for_range(
        layer_range: tuple[int, int],
        placements: List[ModelPlacement],
    ) -> Optional[ModelPlacement]:
        start, end = layer_range
        for placement in placements:
            if placement.layer_range is None:
                continue
            owner_start, owner_end = placement.layer_range
            if owner_start <= start and end <= owner_end:
                return placement
        return None
