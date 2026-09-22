"""运行无 GPU 的多节点分层推理模拟。

示例：
    python -m edge_llm_scheduler.experiments.run_layered_simulation --nodes 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ..backends import LayeredMockEngine, MockKVStore, MockTransport
from ..core import (
    EventBus,
    InferenceRequest,
    ModelManager,
    ModelPlacement,
    ModelSpec,
    Node,
    NodeCapability,
    NodeRole,
    NodeState,
    PipelineReconfigurationCoordinator,
    RequestFlow,
    TaskScheduler,
)
from ..core.node_manager import NodeManager
from ..policies.placement import LayeredPipelinePlacement
from ..policies.reparallelization import CapabilityReparallelization


def make_node(node_id: str, flops: float) -> Node:
    memory = 2 * 1024**3
    return Node(
        node_id=node_id,
        capability=NodeCapability(
            compute_flops=flops,
            memory_total=memory,
            bandwidth=30.0,
            supported_roles={NodeRole.GENERAL},
        ),
        state=NodeState(memory_free=int(memory * 0.9), load=0.0),
    )


async def run(nodes_count: int) -> None:
    bus = EventBus()
    bus.start()
    node_manager = NodeManager(event_bus=bus)
    nodes = [make_node(f"node_{i}", 100.0 + i * 50.0) for i in range(nodes_count)]
    for node in nodes:
        await node_manager.register(node)

    model = ModelSpec(
        model_name="cpu-layered-demo",
        num_layers=12,
        hidden_size=1024,
        total_params_bytes=384 * 1024**2,
        kv_bytes_per_token=4096,
    )
    model_manager = ModelManager(model, node_manager)
    repartition_policy = CapabilityReparallelization()
    placements = await repartition_policy.reconfigure(model, nodes, {})
    pipeline_policy = LayeredPipelinePlacement()
    runtimes = {
        placement.node_id: LayeredMockEngine(
            placement.node_id,
            total_layers=model.num_layers,
            hidden_size=model.hidden_size,
            time_scale=1.0,
        )
        for placement in placements
    }
    controller = PipelineReconfigurationCoordinator(
        node_manager=node_manager,
        model_manager=model_manager,
        storage=MockKVStore(),
        placement_policy=pipeline_policy,
        runtimes=runtimes,
    )
    report = await controller.apply_plan(placements)

    scheduler = TaskScheduler(
        node_manager=node_manager,
        storage=controller.storage,
        transport=MockTransport(bandwidth_mbps=30.0, latency_ms=20.0),
        event_bus=bus,
        placement_policy=pipeline_policy,
        model_manager=model_manager,
        pipeline_controller=controller,
    )
    for node_id, runtime in runtimes.items():
        scheduler.attach_engine(node_id, runtime)

    result = await RequestFlow(scheduler).process(
        InferenceRequest(
            request_id="layered-demo",
            prompt="shared edge prompt " * 8,
            max_tokens=8,
        )
    )
    index = await scheduler.storage.get_index()
    logging.info(
        "result=%s tokens=%d stages=%d kv_blocks=%d locations=%s",
        result.text,
        result.num_tokens,
        len(report.placements),
        len(index),
        index,
    )
    await bus.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run(max(1, args.nodes)))


if __name__ == "__main__":
    main()
