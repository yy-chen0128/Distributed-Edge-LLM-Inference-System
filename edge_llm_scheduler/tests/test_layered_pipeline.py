"""无 GPU 分层 pipeline 与 KV 生命周期测试。"""

from __future__ import annotations

import pytest

from edge_llm_scheduler.backends import LayeredMockEngine, MockKVStore, MockTransport
from edge_llm_scheduler.core import (
    EventBus,
    InferenceRequest,
    ModelPlacement,
    Node,
    NodeCapability,
    NodeState,
    RequestFlow,
    TaskScheduler,
)
from edge_llm_scheduler.core.node_manager import NodeManager
from edge_llm_scheduler.policies.placement import LayeredPipelinePlacement


def make_node(node_id: str) -> Node:
    return Node(
        node_id=node_id,
        capability=NodeCapability(
            compute_flops=100.0,
            memory_total=2 * 1024**3,
            bandwidth=30.0,
        ),
        state=NodeState(memory_free=2 * 1024**3, load=0.0),
    )


@pytest.mark.asyncio
async def test_layered_pipeline_runs_each_stage_and_registers_kv():
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    for node_id in ("a", "b", "c"):
        await nm.register(make_node(node_id))

    placements = [
        ModelPlacement(node_id="a", layer_range=(0, 4)),
        ModelPlacement(node_id="b", layer_range=(4, 8)),
        ModelPlacement(node_id="c", layer_range=(8, 12)),
    ]
    storage = MockKVStore()
    scheduler = TaskScheduler(
        node_manager=nm,
        storage=storage,
        event_bus=bus,
        placement_policy=LayeredPipelinePlacement(placements),
    )
    engines = {
        node_id: LayeredMockEngine(node_id, total_layers=12, time_scale=0.0)
        for node_id in ("a", "b", "c")
    }
    for node_id, engine in engines.items():
        scheduler.attach_engine(node_id, engine)

    result = await RequestFlow(scheduler).process(
        InferenceRequest("layered-1", prompt="hello " * 16, max_tokens=4)
    )
    await bus.stop()

    assert result.error is None
    assert result.text.endswith("c]")
    assert result.num_tokens == 4
    assert all(engine.stage_calls == 1 for engine in engines.values())

    index = await storage.get_index()
    assert len(index) == 3
    locations = {location for locs in index.values() for location in locs}
    assert locations == {"a:gpu", "b:gpu", "c:gpu"}
    blocks = [
        await storage.load(block_hash, location)
        for block_hash, locs in index.items()
        for location in locs
        for block in [await storage.load(block_hash, location)]
    ]
    assert {block.layer_range for block in blocks if block} == {(0, 4), (4, 8), (8, 12)}


@pytest.mark.asyncio
async def test_mock_kv_move_requires_the_declared_source():
    storage = MockKVStore()
    from edge_llm_scheduler.core.types import KVBlock

    await storage.save(KVBlock(1, 16, 100), "a:gpu")
    with pytest.raises(KeyError):
        await storage.move(1, "b:gpu", "c:gpu")


@pytest.mark.asyncio
async def test_layered_pipeline_sends_a_checked_activation_over_transport():
    nm = NodeManager()
    for node_id in ("a", "b"):
        await nm.register(make_node(node_id))

    engines = {
        node_id: LayeredMockEngine(node_id, total_layers=8, time_scale=0.0)
        for node_id in ("a", "b")
    }
    scheduler = TaskScheduler(
        node_manager=nm,
        storage=MockKVStore(),
        transport=MockTransport(bandwidth_mbps=10_000.0, latency_ms=0.0),
        placement_policy=LayeredPipelinePlacement([
            ModelPlacement(node_id="a", layer_range=(0, 4)),
            ModelPlacement(node_id="b", layer_range=(4, 8)),
        ], pipeline_epoch=7),
    )
    for node_id, engine in engines.items():
        scheduler.attach_engine(node_id, engine)

    result = await scheduler.handle_request(
        InferenceRequest("activation-1", prompt="edge activation", max_tokens=2)
    )

    assert result.error is None
    assert len(engines["b"].received_activations) == 1
    activation = engines["b"].received_activations[0]
    assert activation.pipeline_epoch == 7
    assert activation.source_node == "a"
    assert activation.destination_node == "b"
    assert activation.metadata["wire_bytes"] > len(activation.payload)
