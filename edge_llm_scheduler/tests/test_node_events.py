"""验证：节点加入/离开事件流正确分发，触发迁移/放置策略。"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core import (
    Event, EventBus, EventType, InferenceRequest, Node, NodeCapability,
    NodeRole, NodeState, TaskScheduler,
)
from edge_llm_scheduler.core.node_manager import NodeManager
from edge_llm_scheduler.backends import MockEngine, MockKVStore, MockTransport


def make_node(node_id: str, load: float = 0.1) -> Node:
    return Node(
        node_id=node_id,
        capability=NodeCapability(compute_flops=100.0, memory_total=16 * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=12 * 1024**3, load=load),
    )


@pytest.mark.asyncio
async def test_node_join_publishes_event():
    bus = EventBus()
    bus.start()
    seen = []

    async def on_join(ev: Event):
        seen.append(ev.node_id)

    bus.subscribe(EventType.NODE_JOINED, on_join)
    nm = NodeManager(event_bus=bus)
    await nm.register(make_node("new_node"))
    await asyncio.sleep(0.05)
    await bus.stop()

    assert "new_node" in seen


@pytest.mark.asyncio
async def test_node_leave_triggers_migration():
    """节点离开：KV 块应被迁移到剩余节点。"""
    bus = EventBus()
    bus.start()
    storage = MockKVStore()
    transport = MockTransport()

    scheduler = TaskScheduler(node_manager=None, storage=storage, transport=transport, event_bus=bus)
    nm = NodeManager(event_bus=bus)
    scheduler.node_mgr = nm

    # 注册三个节点，各挂引擎
    for nid in ["a", "b", "c"]:
        await nm.register(make_node(nid))
        scheduler.attach_engine(nid, MockEngine(nid))

    # 模拟节点 a 上存了几个 KV 块
    from edge_llm_scheduler.core.types import KVBlock
    for h in [101, 102, 103]:
        await storage.save(KVBlock(block_hash=h, num_tokens=16, byte_size=1000), "a:gpu")

    # 触发节点 a 离开
    await nm.unregister("a", reason="test")
    await asyncio.sleep(0.1)  # 等事件处理

    index = await storage.get_index()
    await bus.stop()

    # a 的 KV 块应已迁走（不再只有 a 持有）
    for h in [101, 102, 103]:
        locs = index.get(h, [])
        assert locs, f"block {h} lost"
        assert not any(l.startswith("a:") for l in locs), f"block {h} still on a"


@pytest.mark.asyncio
async def test_heartbeat_marks_stale_node_left():
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus, heartbeat_timeout_sec=0.05)
    await nm.register(make_node("ghost"))
    await asyncio.sleep(0.12)
    await nm.check_stale_nodes()
    await bus.stop()

    ghost = nm.get("ghost")
    assert ghost is not None and ghost.state.alive is False
