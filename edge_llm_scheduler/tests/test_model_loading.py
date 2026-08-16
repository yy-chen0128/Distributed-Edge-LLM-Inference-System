"""验证：模型装载接口（控制装哪些层/哪些专家/专家进哪级存储）。"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core import (
    EventBus, ExpertSpec, ModelManager, ModelPlacement, ModelSpec,
    Node, NodeCapability, NodeRole, NodeState,
)
from edge_llm_scheduler.core.node_manager import NodeManager


def make_node(node_id: str, mem_gb: int = 16) -> Node:
    return Node(
        node_id=node_id,
        capability=NodeCapability(compute_flops=100.0, memory_total=mem_gb * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=int(mem_gb * 0.8 * 1024**3), load=0.0),
    )


def make_moe_model() -> ModelSpec:
    """16 层，每层 4 个专家的小型 MoE 模型。"""
    experts = [
        ExpertSpec(expert_id=l * 4 + e, layer_id=l, size_bytes=64 * 1024**2)  # 每专家 64MB
        for l in range(16) for e in range(4)
    ]
    return ModelSpec(
        model_name="mock-moe",
        num_layers=16,
        hidden_size=1024,
        num_experts_per_layer=4,
        experts=experts,
        total_params_bytes=64 * 16 * 4 * 1024**2,
        kv_bytes_per_token=2048,
    )


@pytest.mark.asyncio
async def test_load_layers_consumes_memory():
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    node = make_node("n1")
    await nm.register(node)

    model = make_moe_model()
    mm = ModelManager(model=model, node_manager=nm)

    await mm.load_layers("n1", (0, 8))
    p = mm.get_placement("n1")
    assert p is None  # load_layers 不写 placement（那是 load_placement 的事）

    # 检查节点状态更新
    assert nm.get("n1").state.layer_range == (0, 8)
    await bus.stop()


@pytest.mark.asyncio
async def test_load_experts_to_tiers():
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    await nm.register(make_node("n1"))

    model = make_moe_model()
    mm = ModelManager(model=model, node_manager=nm)

    await mm.load_experts("n1", [0, 1, 2], tier="gpu")
    await mm.load_experts("n1", [3, 4], tier="cpu")

    node = nm.get("n1")
    assert node.state.expert_ids == {0, 1, 2}
    assert node.state.expert_ids_cpu == {3, 4}
    await bus.stop()


@pytest.mark.asyncio
async def test_load_placement_with_insufficient_memory_raises():
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    await nm.register(make_node("tiny", mem_gb=1))  # 1GB 太小

    model = make_moe_model()
    mm = ModelManager(model=model, node_manager=nm)

    placement = ModelPlacement(
        node_id="tiny",
        layer_range=(0, 16),          # 全部层
        experts_gpu=list(range(64)),   # 全部专家
    )
    with pytest.raises(MemoryError):
        await mm.load_placement(placement)
    await bus.stop()


@pytest.mark.asyncio
async def test_load_placement_updates_state():
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    await nm.register(make_node("n1"))

    model = make_moe_model()
    mm = ModelManager(model=model, node_manager=nm)

    placement = ModelPlacement(node_id="n1", layer_range=(0, 4), experts_gpu=[0, 1], experts_cpu=[2])
    await mm.load_placement(placement)

    p = mm.get_placement("n1")
    assert p is not None
    assert p.layer_range == (0, 4)
    assert p.experts_gpu == [0, 1]
    assert p.experts_cpu == [2]
    assert mm.node_has_layer("n1", 2)
    assert not mm.node_has_layer("n1", 4)
    assert mm.node_has_expert("n1", 1, tier="gpu")
    assert mm.node_has_expert("n1", 2, tier="cpu")
    await bus.stop()
