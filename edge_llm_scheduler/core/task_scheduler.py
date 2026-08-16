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
from typing import Callable, Dict, List, Optional

from .event_bus import EventBus
from .node_manager import NodeManager
from .storage import KVStore
from .transport import Transport
from .types import (
    Event, EventType, GenerationResult, InferenceRequest, KVBlock, Node, Task,
)
from ..policies.placement import PlacementPolicy, DefaultPlacement
from ..policies.migration import MigrationPolicy, DefaultMigration
from ..policies.recovery import RecoveryPolicy, DefaultRecovery

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
    ) -> None:
        self.node_mgr = node_manager
        self.storage = storage
        self.transport = transport
        self.event_bus = event_bus
        self.placement = placement_policy or DefaultPlacement()
        self.migration = migration_policy or DefaultMigration()
        self.recovery = recovery_policy or DefaultRecovery()

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
        results = await asyncio.gather(*(self._run_task(t) for t in tasks))
        # 汇总（当前取第一个成功结果；多 Task 拼装后续补充）
        result = next((r for r in results if r is not None and r.error is None), results[0])
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
        return await self.placement.place(
            model=None, nodes=nodes, request=req, storage=self.storage, hit_tokens=hit
        )

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
            # 结果里的新 KV 块登记进 KVStore
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
        node_id = event.node_id
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

    async def _on_node_joined(self, event: Event) -> None:
        """节点加入：放置策略决定给新节点分配什么工作。"""
        # 默认实现：新节点由上层（ReparallelizationPolicy 或手动）安排
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
