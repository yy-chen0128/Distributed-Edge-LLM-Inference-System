"""验证：完整请求生命周期（mock 全套，无硬件）。

节点注册 → 模型装载 → 请求到达 → 路由 → 下发引擎 → 结果回收 → KV 落存储。
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core import (
    EventBus, InferenceRequest, Node, NodeCapability, NodeRole, NodeState,
    TaskScheduler, RequestFlow,
)
from edge_llm_scheduler.backends import MockEngine, MockKVStore, MockTransport


def make_node(node_id: str, mem_gb: int = 16, load: float = 0.1) -> Node:
    return Node(
        node_id=node_id,
        capability=NodeCapability(
            compute_flops=100.0,
            memory_total=mem_gb * 1024**3,
            bandwidth=100.0,
            supported_roles={NodeRole.GENERAL},
        ),
        state=NodeState(memory_free=int(mem_gb * 0.8 * 1024**3), load=load),
    )


@pytest.mark.asyncio
async def test_full_request_lifecycle():
    bus = EventBus()
    bus.start()
    storage = MockKVStore()
    transport = MockTransport()

    scheduler = TaskScheduler(
        node_manager=None,  # 占位，下面手动构造完整实例
        storage=storage,
        transport=transport,
        event_bus=bus,
    )
    # 构造 node_manager 和引擎
    from edge_llm_scheduler.core.node_manager import NodeManager
    nm = NodeManager(event_bus=bus)
    scheduler.node_mgr = nm

    n1, n2 = make_node("node_a", load=0.1), make_node("node_b", load=0.4)
    await nm.register(n1)
    await nm.register(n2)
    scheduler.attach_engine("node_a", MockEngine("node_a", tokens_per_sec=20))
    scheduler.attach_engine("node_b", MockEngine("node_b", tokens_per_sec=10))

    flow = RequestFlow(scheduler)
    req = InferenceRequest(request_id="req-1", prompt="hello world", max_tokens=16)

    result = await flow.process(req)
    await bus.stop()

    assert result.error is None
    assert result.num_tokens > 0
    assert result.latency_ms > 0
    assert len(result.kv_block_ids) > 0


@pytest.mark.asyncio
async def test_routes_to_lightest_node():
    bus = EventBus()
    bus.start()
    storage = MockKVStore()
    scheduler = TaskScheduler(
        node_manager=None, storage=storage, event_bus=bus
    )
    from edge_llm_scheduler.core.node_manager import NodeManager
    nm = NodeManager(event_bus=bus)
    scheduler.node_mgr = nm

    await nm.register(make_node("heavy", load=0.9))
    await nm.register(make_node("light", load=0.1))
    scheduler.attach_engine("heavy", MockEngine("heavy", tokens_per_sec=5))
    scheduler.attach_engine("light", MockEngine("light", tokens_per_sec=20))

    req = InferenceRequest(request_id="req-2", prompt="test prompt")
    tasks = await scheduler.route_request(req)
    await bus.stop()

    assert tasks and tasks[0].node_id == "light"


@pytest.mark.asyncio
async def test_no_healthy_node_returns_error():
    bus = EventBus()
    bus.start()
    scheduler = TaskScheduler(node_manager=None, storage=MockKVStore(), event_bus=bus)
    from edge_llm_scheduler.core.node_manager import NodeManager
    nm = NodeManager(event_bus=bus)
    scheduler.node_mgr = nm

    await nm.register(make_node("solo", load=1.0))
    req = InferenceRequest(request_id="req-3", prompt="x")
    result = await scheduler.handle_request(req)
    await bus.stop()

    assert result.error is not None
