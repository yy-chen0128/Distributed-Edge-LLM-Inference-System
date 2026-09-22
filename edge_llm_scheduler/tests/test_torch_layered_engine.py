"""Torch stage runtime tests for tensor activation flow and epoch unload."""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from edge_llm_scheduler.backends import MockKVStore, MockTransport, TorchLayeredEngine
from edge_llm_scheduler.core import (
    InferenceRequest,
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


def make_node(node_id: str) -> Node:
    memory = 2 * 1024**3
    return Node(
        node_id=node_id,
        capability=NodeCapability(compute_flops=100.0, memory_total=memory),
        state=NodeState(memory_free=memory),
    )


async def make_pipeline(node_ids: tuple[str, ...], placements: list[ModelPlacement]):
    nodes = NodeManager()
    for node_id in node_ids:
        await nodes.register(make_node(node_id))
    model = ModelSpec(
        model_name="toy-tensor",
        num_layers=6,
        hidden_size=16,
        total_params_bytes=6 * 16 * 16 * 4,
    )
    storage = MockKVStore()
    policy = LayeredPipelinePlacement()
    runtimes = {
        node_id: TorchLayeredEngine(
            node_id,
            total_layers=model.num_layers,
            hidden_size=model.hidden_size,
            vocab_size=128,
            seed=123,
        )
        for node_id in node_ids
    }
    controller = PipelineReconfigurationCoordinator(
        nodes,
        ModelManager(model, nodes),
        storage,
        policy,
        runtimes,
    )
    await controller.apply_plan(placements)
    scheduler = TaskScheduler(
        node_manager=nodes,
        storage=storage,
        transport=MockTransport(bandwidth_mbps=10_000.0, latency_ms=0.0),
        placement_policy=policy,
        pipeline_controller=controller,
    )
    for node_id, runtime in runtimes.items():
        scheduler.attach_engine(node_id, runtime)
    return scheduler, controller, storage, runtimes


def test_torch_runtime_auto_uses_cpu_when_no_ascend_npu_is_available():
    if hasattr(torch, "npu") and torch.npu.is_available():
        pytest.skip("Ascend NPU is available in this environment")

    runtime = TorchLayeredEngine(
        "npu-auto",
        total_layers=2,
        hidden_size=4,
        vocab_size=16,
        device="auto",
    )

    assert runtime.device.type == "cpu"


def test_torch_runtime_rejects_explicit_npu_without_torch_npu():
    if importlib.util.find_spec("torch_npu") is not None:
        pytest.skip("torch-npu is installed in this environment")

    with pytest.raises(RuntimeError, match="torch-npu"):
        TorchLayeredEngine(
            "npu-required",
            total_layers=2,
            hidden_size=4,
            vocab_size=16,
            device="npu",
        )


@pytest.mark.asyncio
async def test_torch_pipeline_transfers_real_hidden_state_between_stages():
    scheduler, _, storage, runtimes = await make_pipeline(
        ("a", "b", "c"),
        [
            ModelPlacement(node_id="a", layer_range=(0, 2)),
            ModelPlacement(node_id="b", layer_range=(2, 4)),
            ModelPlacement(node_id="c", layer_range=(4, 6)),
        ],
    )

    result = await scheduler.handle_request(
        InferenceRequest("torch-pipe-1", prompt="edge tensor flow", max_tokens=3)
    )

    assert result.error is None
    assert result.text.startswith("tok")
    assert result.num_tokens == 3
    assert all(runtime.stage_calls == 1 for runtime in runtimes.values())
    assert len(runtimes["b"].received_activations) == 1
    assert len(runtimes["c"].received_activations) == 1
    activation = runtimes["b"].received_activations[0]
    assert activation.metadata["tensor_kind"] == "torch.hidden_states"
    assert activation.metadata["shape"][1] == 16
    assert activation.metadata["wire_bytes"] > len(activation.payload)

    index = await storage.get_index()
    blocks = [
        await storage.load(block_hash, location)
        for block_hash, locations in index.items()
        for location in locations
    ]
    assert {block.layer_range for block in blocks if block is not None} == {
        (0, 2),
        (2, 4),
        (4, 6),
    }


@pytest.mark.asyncio
async def test_torch_runtime_prepares_activates_and_unloads_epochs():
    _, controller, _, runtimes = await make_pipeline(
        ("a", "b", "c"),
        [
            ModelPlacement(node_id="a", layer_range=(0, 3)),
            ModelPlacement(node_id="b", layer_range=(3, 6)),
        ],
    )

    assert runtimes["a"].active_epoch == 1
    assert runtimes["a"].loaded_layer_ranges == {1: (0, 3)}

    await controller.apply_plan([
        ModelPlacement(node_id="c", layer_range=(0, 3)),
        ModelPlacement(node_id="b", layer_range=(3, 6)),
    ])
    await controller.drain_old_epoch(1)

    assert runtimes["a"].active_epoch is None
    assert runtimes["a"].loaded_layer_ranges == {}
    assert runtimes["c"].active_epoch == 2
    assert runtimes["c"].loaded_layer_ranges == {2: (0, 3)}
