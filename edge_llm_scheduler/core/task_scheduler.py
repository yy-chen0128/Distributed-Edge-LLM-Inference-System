"""任务调度编排。

机制：
- 路由请求 → 生成 Task（交给 PlacementPolicy 决定去哪个节点）
- 下发任务到节点引擎
- 回收结果
- 节点离开/加入事件 → 触发策略（迁移/重并行化）

策略注入：PlacementPolicy / MigrationPolicy / ReparallelizationPolicy / RecoveryPolicy
都是可插拔的。框架只编排流程，具体算法在 policies/ 里。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from .event_bus import EventBus
from .node_manager import NodeManager
from .storage import KVStore
from .transport import Transport
from .types import (
    ActivationEnvelope, Event, EventType, GenerationResult, InferenceRequest, KVBlock, Node, Task,
)
from ..policies.placement import PlacementPolicy, DefaultPlacement
from ..policies.migration import MigrationPolicy, DefaultMigration
from ..policies.recovery import RecoveryPolicy, DefaultRecovery

if TYPE_CHECKING:
    from .pipeline_controller import PipelineReconfigurationCoordinator

logger = logging.getLogger(__name__)


class Engine:
    """节点上的推理引擎抽象（mock 或 vLLM 都实现它）。"""

    async def generate(self, task: Task) -> GenerationResult:
        raise NotImplementedError


class TaskScheduler:
    def __init__(
        self,
        node_manager: NodeManager,
        storage: KVStore,
        transport: Optional[Transport] = None,
        event_bus: Optional[EventBus] = None,
        placement_policy: Optional[PlacementPolicy] = None,
        migration_policy: Optional[MigrationPolicy] = None,
        recovery_policy: Optional[RecoveryPolicy] = None,
        model_manager=None,
        pipeline_controller: Optional["PipelineReconfigurationCoordinator"] = None,
    ) -> None:
        self.node_mgr = node_manager
        self.storage = storage
        self.transport = transport
        self.event_bus = event_bus
        self.placement = placement_policy or DefaultPlacement()
        self.migration = migration_policy or DefaultMigration()
        self.recovery = recovery_policy or DefaultRecovery()
        self.model_manager = model_manager
        self.pipeline_controller = pipeline_controller
        # EventBus 是异步队列；调度器创建前已入队的节点事件属于历史状态，
        # 不应在控制器挂载后重新触发一次拓扑变更。
        self._event_subscription_time = time.time()

        # node_id -> Engine（真实引擎或 mock）
        self._engines: Dict[str, Engine] = {}
        # task_id -> Task
        self._tasks: Dict[str, Task] = {}
        # request_id -> 结果
        self._results: Dict[str, GenerationResult] = {}
        # 内存块：block_hash -> KVBlock（从 KVStore 取回的引用）
        self._kv_blocks: Dict[int, KVBlock] = {}

        if self.event_bus:
            self.event_bus.subscribe_many({
                EventType.NODE_LEFT: self._on_node_left,
                EventType.NODE_JOINED: self._on_node_joined,
            })

    # ---------- 引擎注册 ----------

    def attach_engine(self, node_id: str, engine: Engine) -> None:
        self._engines[node_id] = engine

    # ---------- 请求处理 ----------

    async def handle_request(self, req: InferenceRequest) -> GenerationResult:
        """请求入口：路由 → 下发 → 回收。"""
        if self.event_bus:
            await self.event_bus.publish(
                Event(type=EventType.REQUEST_ARRIVED, request_id=req.request_id, payload=req, timestamp=time.time())
            )
        tasks = await self.route_request(req)
        if not tasks:
            return GenerationResult(request_id=req.request_id, error="no available node")
        if len(tasks) == 1:
            results = [await self._run_task(tasks[0])]
        else:
            # 多个任务表示一个层级 pipeline，而不是多个独立副本。
            # stage 必须串行推进：后一个 stage 消费前一个 stage 的 activation。
            results = [await self._run_pipeline(tasks)]
        result = next((r for r in results if r is not None and r.error is None), results[0])
        if result is None:
            result = GenerationResult(request_id=req.request_id, error="task execution failed")
        self._results[req.request_id] = result
        return result

    async def route_request(self, req: InferenceRequest) -> List[Task]:
        """路由：用放置策略决定请求去哪（哪些节点、哪些层）。"""
        nodes = self.node_mgr.healthy_nodes()
        if not nodes:
            logger.warning(f"no healthy node for request {req.request_id}")
            return []
        # 查询 KV 前缀命中（缓存感知路由的依据）
        prefix_hashes = self._hashes_from_prompt(req.prompt)
        hit = await self.storage.lookup(prefix_hashes)
        tasks = await self.placement.place(
            model=None, nodes=nodes, request=req, storage=self.storage, hit_tokens=hit
        )
        # 让所有后端都使用正式 Task 字段；兼容旧的第三方 PlacementPolicy。
        for index, task in enumerate(tasks):
            task.prompt = req.prompt
            task.max_tokens = req.max_tokens
            task.prompt_len = self._prompt_len(req.prompt)
            task.hit_tokens = hit
            task.stage_index = index
            task.stage_count = len(tasks)
        return tasks

    async def _run_pipeline(self, tasks: List[Task]) -> Optional[GenerationResult]:
        """按层区间顺序执行一个模拟/真实 stage pipeline。"""
        ordered = sorted(tasks, key=lambda t: (
            t.layer_range[0] if t.layer_range else t.stage_index,
            t.stage_index,
        ))
        activation: Optional[ActivationEnvelope] = None
        previous_node = None
        final_result: Optional[GenerationResult] = None
        for index, task in enumerate(ordered):
            task.stage_index = index
            task.stage_count = len(ordered)
            task.activation = activation
            if activation is not None and previous_node != task.node_id:
                activation = await self._deliver_activation(activation, task.node_id)
                task.activation = activation
            result = await self._run_task(task)
            if result is None or result.error is not None:
                return result
            activation = result.activation
            previous_node = task.node_id
            final_result = result
        return final_result

    async def _deliver_activation(
        self,
        activation: ActivationEnvelope,
        destination_node: str,
    ) -> ActivationEnvelope:
        """把上一 stage 的 hidden states 作为受校验的数据面消息交给下一 stage。"""
        delivery = activation.for_destination(destination_node)
        if self.transport is None:
            return delivery

        started = time.perf_counter()
        raw = delivery.to_bytes()
        await self.transport.push(raw, destination_node, delivery.tag)
        received = await self.transport.pull(destination_node, delivery.tag)
        if received is None:
            raise RuntimeError(
                f"activation delivery failed for request {delivery.request_id}: "
                f"{delivery.source_node} -> {destination_node}"
            )
        delivered = ActivationEnvelope.from_bytes(received)
        if delivered.destination_node != destination_node:
            raise RuntimeError("activation destination changed during transport")
        delivered.metadata["transfer_ms"] = (time.perf_counter() - started) * 1000.0
        delivered.metadata["wire_bytes"] = len(raw)
        return delivered

    async def _run_task(self, task: Task) -> Optional[GenerationResult]:
        """下发任务到节点引擎并回收结果。"""
        engine = self._engines.get(task.node_id)
        if engine is None:
            logger.error(f"no engine on node {task.node_id}")
            task.mark("failed")
            return None
        task.mark("running")
        self._tasks[task.task_id] = task
        node = self.node_mgr.get(task.node_id)
        if node:
            node.state.pending_tasks.add(task.task_id)
        try:
            result = await engine.generate(task)
            task.mark("done")
            task.finish_time = time.time()
            # 结果里的新 KV 块登记进 KVStore。真实后端可以只返回 ids，
            # CPU 模拟器则返回带位置/层区间的完整块。
            for block in result.kv_blocks:
                self._kv_blocks[block.block_hash] = block
                if block.block_hash not in result.kv_block_ids:
                    result.kv_block_ids.append(block.block_hash)
                await self.storage.save(block, f"{task.node_id}:gpu")
            for bid in result.kv_block_ids:
                block = self._kv_blocks.get(bid)
                if block:
                    await self.storage.save(block, f"{task.node_id}:gpu")
            if self.event_bus:
                await self.event_bus.publish(
                    Event(type=EventType.TASK_DONE, request_id=task.request_id,
                          node_id=task.node_id, timestamp=time.time(), payload=result)
                )
            return result
        except Exception as e:
            task.mark("interrupted")
            logger.warning(f"task {task.task_id} interrupted on {task.node_id}: {e}")
            if self.event_bus:
                await self.event_bus.publish(
                    Event(type=EventType.TASK_INTERRUPTED, request_id=task.request_id,
                          node_id=task.node_id, timestamp=time.time(), payload=task)
                )
            # 中断恢复：让 recovery 策略决定（默认：重新调度）
            retry = await self.recovery.recover(task)
            if retry is not None:
                return await self._run_task(retry)
            return None
        finally:
            if node:
                node.state.pending_tasks.discard(task.task_id)

    # ---------- 节点事件 ----------

    async def _on_node_left(self, event: Event) -> None:
        """节点离开：迁移策略决定该节点上的 KV/参数去向。"""
        if event.timestamp and event.timestamp < self._event_subscription_time:
            return
        node_id = event.node_id
        if (
            self.pipeline_controller is not None
            and self.pipeline_controller.current_plan is not None
            and node_id not in {p.node_id for p in self.pipeline_controller.current_plan.placements}
        ):
            return
        node = self.node_mgr.get(node_id)
        if node is None:
            return
        # 收集该节点持有的 KV 块（从 KVStore 索引）
        index = await self.storage.get_index()
        node_blocks = [bid for bid, locs in index.items() if any(l.startswith(f"{node_id}:") for l in locs)]
        # 中断该节点的任务 → 交给 recovery
        for tid in list(node.state.pending_tasks):
            task = self._tasks.get(tid)
            if task:
                retry = await self.recovery.recover(task)
                if retry is not None:
                    await self._run_task(retry)
        # 迁移计划：决定哪些 KV 块传走、传哪（可用目标 = 剩余健康节点）
        remaining = [n for n in self.node_mgr.alive_nodes() if n.node_id != node_id]
        plan = await self.migration.decide(
            node_id=node_id, kv_block_ids=node_blocks, storage=self.storage,
            available_nodes=remaining,
        )
        await self.migration.execute(plan, storage=self.storage, transport=self.transport)
        if self.pipeline_controller is not None:
            await self.pipeline_controller.reconfigure_for_topology()

    async def _on_node_joined(self, event: Event) -> None:
        """节点加入：放置策略决定给新节点分配什么工作。"""
        if event.timestamp and event.timestamp < self._event_subscription_time:
            return
        if self.pipeline_controller is not None:
            if (
                self.pipeline_controller.current_plan is not None
                and event.node_id in {p.node_id for p in self.pipeline_controller.current_plan.placements}
            ):
                return
            await self.pipeline_controller.reconfigure_for_topology()
            return
        logger.info(f"node joined, waiting for placement decision: {event.node_id}")

    # ---------- 工具 ----------

    def _hashes_from_prompt(self, prompt) -> List[int]:
        """把 prompt 切成块哈希（mock 用简单哈希；真实用 LMCache 语义）。"""
        if prompt is None:
            return []
        if isinstance(prompt, str):
            tokens = list(range(len(prompt)))   # 简化：字符串按字符
        else:
            tokens = list(prompt)
        block_size = 16
        blocks = [tokens[i:i+block_size] for i in range(0, len(tokens), block_size)]
        return [hash(tuple(b)) for b in blocks]

    @staticmethod
    def _prompt_len(prompt) -> int:
        if prompt is None:
            return 0
        return len(prompt)
