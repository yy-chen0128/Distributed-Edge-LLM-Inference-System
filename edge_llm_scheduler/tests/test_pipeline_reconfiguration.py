"""epoch 化重分层与 KV 副本测试（纯 CPU）。"""

from __future__ import annotations

import pytest

from edge_llm_scheduler.backends import LayeredMockEngine, MockKVStore
from edge_llm_scheduler.core import (
    EventBus,
    KVBlock,
    ModelManager,
    ModelPlacement,
    ModelSpec,
    Node,
    NodeCapability,
    NodeState,
    PipelineReconfigurationCoordinator,
    TaskScheduler,
)
from edge_llm_scheduler.core.node_manager import NodeManager
from edge_llm_scheduler.policies.placement import LayeredPipelinePlacement


def make_node(node_id: str, flops: float = 100.0) -> Node:
    memory = 2 * 1024**3
    return Node(
        node_id=node_id,
        capability=NodeCapability(compute_flops=flops, memory_total=memory),
        state=NodeState(memory_free=memory),
    )


@pytest.mark.asyncio
async def test_reconfiguration_prepares_new_epoch_and_copies_kv_by_layer_range():
    nodes = NodeManager()
    for node_id in ("a", "b", "c"):
        await nodes.register(make_node(node_id))
    model = ModelSpec(
        model_name="tiny",
        num_layers=8,
        hidden_size=64,
        total_params_bytes=800 * 1024**2,
    )
    storage = MockKVStore()
    policy = LayeredPipelinePlacement()
    runtimes = {
        node_id: LayeredMockEngine(node_id, total_layers=8, time_scale=0.0)
        for node_id in ("a", "b", "c")
    }
    controller = PipelineReconfigurationCoordinator(
        nodes,
        ModelManager(model, nodes),
        storage,
        policy,
        runtimes,
    )

    initial = await controller.apply_plan([
        ModelPlacement(node_id="a", layer_range=(0, 4)),
        ModelPlacement(node_id="b", layer_range=(4, 8)),
    ])
    assert initial.pipeline_epoch == 1
    await storage.save(KVBlock(10, 16, 128, layer_range=(0, 4), data=b"left"), "a:gpu")
    await storage.save(KVBlock(11, 16, 128, layer_range=(4, 8), data=b"right"), "b:gpu")

    report = await controller.apply_plan([
        ModelPlacement(node_id="c", layer_range=(0, 4)),
        ModelPlacement(node_id="b", layer_range=(4, 8)),
    ])

    assert report.pipeline_epoch == 2
    assert policy.pipeline_epoch == 2
    assert runtimes["c"]._active_epoch == 2
    assert await storage.load(10, "c:gpu") is not None
    assert await storage.load(10, "a:gpu") is not None  # drain 前保留旧 epoch 副本
    assert [(copy.block_hash, copy.destination) for copy in report.kv_copies] == [(10, "c:gpu")]


@pytest.mark.asyncio
async def test_topology_reconfiguration_uses_heterogeneous_compute_weights():
    nodes = NodeManager()
    for node_id, flops in (("a", 1.0), ("b", 3.0)):
        await nodes.register(make_node(node_id, flops))
    model = ModelSpec("tiny", num_layers=8, hidden_size=64, total_params_bytes=80 * 1024**2)
    policy = LayeredPipelinePlacement()
    controller = PipelineReconfigurationCoordinator(
        nodes,
        ModelManager(model, nodes),
        MockKVStore(),
        policy,
        {
            "a": LayeredMockEngine("a", total_layers=8, time_scale=0.0),
            "b": LayeredMockEngine("b", total_layers=8, time_scale=0.0),
        },
    )

    report = await controller.reconfigure_for_topology()

    assert report.pipeline_epoch == 1
    assert [placement.layer_range for placement in report.placements] == [(0, 2), (2, 8)]


@pytest.mark.asyncio
async def test_node_join_and_leave_trigger_new_pipeline_epochs():
    bus = EventBus()
    bus.start()
    nodes = NodeManager(event_bus=bus)
    for node_id in ("a", "b"):
        await nodes.register(make_node(node_id))
    model = ModelSpec("tiny", num_layers=8, hidden_size=64, total_params_bytes=80 * 1024**2)
    policy = LayeredPipelinePlacement()
    runtimes = {
        node_id: LayeredMockEngine(node_id, total_layers=8, time_scale=0.0)
        for node_id in ("a", "b", "c")
    }
    controller = PipelineReconfigurationCoordinator(
        nodes,
        ModelManager(model, nodes),
        MockKVStore(),
        policy,
        runtimes,
    )
    await controller.apply_plan([
        ModelPlacement(node_id="a", layer_range=(0, 4)),
        ModelPlacement(node_id="b", layer_range=(4, 8)),
    ])
    scheduler = TaskScheduler(
        node_manager=nodes,
        storage=controller.storage,
        event_bus=bus,
        placement_policy=policy,
        pipeline_controller=controller,
    )

    await nodes.register(make_node("c", flops=300.0))
    await bus._queue.join()
    assert controller.current_plan is not None
    assert controller.current_plan.pipeline_epoch == 2
    assert {p.node_id for p in controller.current_plan.placements} == {"a", "b", "c"}

    await nodes.unregister("b")
    await bus._queue.join()
    assert controller.current_plan.pipeline_epoch == 3
    assert {p.node_id for p in controller.current_plan.placements} == {"a", "c"}
    await bus.stop()
