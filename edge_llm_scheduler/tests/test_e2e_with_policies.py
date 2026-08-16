"""端到端：E2Placement + PriorityMigration 注入 TaskScheduler 的完整流程。

验证：
- E2 路由在真实调度流程中生效（有缓存 → 命中节点）
- 节点离开时 PriorityMigration 按价值迁移
- 组合使用不破坏原有生命周期
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core import (
    EventBus, InferenceRequest, KVBlock, Node, NodeCapability, NodeRole,
    NodeState, TaskScheduler, RequestFlow,
)
from edge_llm_scheduler.core.node_manager import NodeManager
from edge_llm_scheduler.backends import MockEngine, MockKVStore
from edge_llm_scheduler.policies.placement import E2Placement
from edge_llm_scheduler.policies.migration import PriorityMigration


def make_node(nid: str, load: float) -> Node:
    return Node(
        node_id=nid,
        capability=NodeCapability(compute_flops=100.0, memory_total=16 * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=12 * 1024**3, load=load),
    )


def make_e2(storage) -> E2Placement:
    def hash_fn(prompt, block_size):
        n = len(prompt) // block_size + (1 if len(prompt) % block_size else 0)
        return [100 + i for i in range(n)]
    return E2Placement(storage=storage, hash_fn=hash_fn)


@pytest.mark.asyncio
async def test_e2_routes_to_cache_holder_in_full_flow():
    """完整流程：node_a 有缓存 → E2 路由命中 node_a。"""
    bus = EventBus()
    bus.start()
    storage = MockKVStore()
    nm = NodeManager(event_bus=bus)
    await nm.register(make_node("node_a", load=0.8))
    await nm.register(make_node("node_b", load=0.1))

    scheduler = TaskScheduler(
        node_manager=nm, storage=storage, event_bus=bus,
        placement_policy=make_e2(storage),
    )
    scheduler.attach_engine("node_a", MockEngine("node_a"))
    scheduler.attach_engine("node_b", MockEngine("node_b"))

    # node_a 持有请求前缀缓存
    for h in [100, 101, 102]:
        await storage.save(KVBlock(block_hash=h, num_tokens=16, byte_size=1000), "node_a:gpu")

    flow = RequestFlow(scheduler)
    req = InferenceRequest(request_id="req-e2", prompt="a" * 64, max_tokens=16)
    result = await flow.process(req)
    await bus.stop()

    assert result.error is None
    # 验证任务真的落在 node_a（exploit）
    assert scheduler._tasks and any(t.node_id == "node_a" for t in scheduler._tasks.values())


@pytest.mark.asyncio
async def test_priority_migration_on_node_left():
    """节点离开：PriorityMigration 按价值迁走高价值块，低价值丢。"""
    bus = EventBus()
    bus.start()
    storage = MockKVStore()
    nm = NodeManager(event_bus=bus)
    for nid in ["a", "b"]:
        await nm.register(make_node(nid, load=0.1))

    scheduler = TaskScheduler(
        node_manager=nm, storage=storage, event_bus=bus,
        migration_policy=PriorityMigration(deadline_ms=0.01),  # 只够传 1 块
    )
    scheduler.attach_engine("a", MockEngine("a"))
    scheduler.attach_engine("b", MockEngine("b"))

    # node a 上两块：高价值 + 低价值
    await storage.save(KVBlock(block_hash=1, num_tokens=16, byte_size=1000,
                               reuse_count=100, prefill_time_ms=50.0), "a:gpu")
    await storage.save(KVBlock(block_hash=2, num_tokens=16, byte_size=1000,
                               reuse_count=1, prefill_time_ms=1.0), "a:gpu")

    await nm.unregister("a", reason="test")
    await asyncio.sleep(0.1)   # 等事件处理（迁移）
    await bus.stop()

    index = await storage.get_index()
    # 高价值块 1 → 迁到 b；低价值块 2 → 被丢
    locs1 = index.get(1, [])
    assert any(l.startswith("b:") for l in locs1), "high-value block 1 not migrated"
    assert 2 not in index, "low-value block 2 should be dropped"
