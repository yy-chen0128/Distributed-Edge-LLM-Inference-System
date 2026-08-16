"""测试：CapabilityPlacement 按节点能力加权路由。

验证：算力强的节点在负载相同（或相近）时优先被选择。
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core.types import InferenceRequest, Node, NodeCapability, NodeState
from edge_llm_scheduler.policies.placement import CapabilityPlacement


def make_node(nid: str, flops: float, load: float) -> Node:
    return Node(
        node_id=nid,
        capability=NodeCapability(compute_flops=flops, memory_total=16 * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=12 * 1024**3, load=load),
    )


def make_req() -> InferenceRequest:
    return InferenceRequest(request_id="req-cap", prompt="hello", max_tokens=16)


@pytest.mark.asyncio
async def test_strong_node_preferred_at_same_load():
    """相同负载 → 算力强的优先。"""
    nodes = [make_node("weak", flops=100.0, load=0.5),
             make_node("strong", flops=200.0, load=0.5)]
    policy = CapabilityPlacement()
    tasks = await policy.place(model=None, nodes=nodes, request=make_req(), storage=None)
    assert tasks and tasks[0].node_id == "strong"


@pytest.mark.asyncio
async def test_strong_node_tolerates_higher_load():
    """算力 2× 的节点，负载略高仍优先（有效负载 = 负载/能力权重）。"""
    # strong 负载 0.6，weak 负载 0.4；strong 能力 2× → 有效负载 0.3 vs 0.8
    nodes = [make_node("weak", flops=100.0, load=0.4),
             make_node("strong", flops=200.0, load=0.6)]
    policy = CapabilityPlacement()
    tasks = await policy.place(model=None, nodes=nodes, request=make_req(), storage=None)
    assert tasks and tasks[0].node_id == "strong"


@pytest.mark.asyncio
async def test_dead_node_skipped():
    nodes = [make_node("dead", flops=100.0, load=0.0), make_node("alive", flops=50.0, load=0.9)]
    nodes[0].state.alive = False
    policy = CapabilityPlacement()
    tasks = await policy.place(model=None, nodes=nodes, request=make_req(), storage=None)
    assert tasks and tasks[0].node_id == "alive"
