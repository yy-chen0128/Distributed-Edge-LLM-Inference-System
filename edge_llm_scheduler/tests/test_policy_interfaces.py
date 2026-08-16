"""验证：策略是可插拔的（框架的核心设计点）。

自定义一个策略实现替换默认的，确认框架能接受并正确调用它。
这证明"复杂算法后续补充时，只需实现接口，不用改框架"。
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core import (
    EventBus, InferenceRequest, KVBlock, Node, NodeCapability, NodeState,
    Task, TaskScheduler,
)
from edge_llm_scheduler.core.node_manager import NodeManager
from edge_llm_scheduler.policies.placement import PlacementPolicy
from edge_llm_scheduler.backends import MockEngine, MockKVStore


def make_node(nid: str, load: float) -> Node:
    return Node(
        node_id=nid,
        capability=NodeCapability(compute_flops=100.0, memory_total=16 * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=12 * 1024**3, load=load),
    )


class PreferNodeAPolicy(PlacementPolicy):
    """自定义策略：永远路由到 node_a（验证可插拔）。"""

    async def place(self, model, nodes, request, storage=None, hit_tokens=0):
        return [
            Task(task_id="custom-task", request_id=request.request_id, node_id="node_a", status="pending")
        ]


@pytest.mark.asyncio
async def test_custom_policy_is_used():
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    await nm.register(make_node("node_a", load=0.9))   # 默认策略会避开它
    await nm.register(make_node("node_b", load=0.1))

    scheduler = TaskScheduler(
        node_manager=nm,
        storage=MockKVStore(),
        event_bus=bus,
        placement_policy=PreferNodeAPolicy(),  # 注入自定义策略
    )
    scheduler.attach_engine("node_a", MockEngine("node_a"))
    scheduler.attach_engine("node_b", MockEngine("node_b"))

    req = InferenceRequest(request_id="req-p", prompt="test")
    tasks = await scheduler.route_request(req)
    await bus.stop()

    # 自定义策略生效：路由到 node_a（尽管它负载高）
    assert tasks and tasks[0].node_id == "node_a"
    assert tasks[0].task_id == "custom-task"


@pytest.mark.asyncio
async def test_default_policy_still_works_without_injection():
    """不注入时用默认策略（回归测试）。"""
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    await nm.register(make_node("node_a", load=0.9))
    await nm.register(make_node("node_b", load=0.1))

    scheduler = TaskScheduler(node_manager=nm, storage=MockKVStore(), event_bus=bus)
    scheduler.attach_engine("node_a", MockEngine("node_a"))
    scheduler.attach_engine("node_b", MockEngine("node_b"))

    req = InferenceRequest(request_id="req-d", prompt="test")
    tasks = await scheduler.route_request(req)
    await bus.stop()

    # 默认策略：路由到负载最轻的 node_b
    assert tasks and tasks[0].node_id == "node_b"
