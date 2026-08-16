"""模拟实验测试：固化"E2 命中率优于 Default"的结果，防回归。

用合成的多前缀负载（Preble 风格）验证：
- E2 缓存感知路由命中率显著高于 Default
- E2 负载分布更均衡
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.backends import MockKVStore
from edge_llm_scheduler.backends.datasets import DatasetLoader
from edge_llm_scheduler.core import (
    EventBus, InferenceRequest, KVBlock, Node, NodeCapability, NodeRole, NodeState,
    TaskScheduler,
)
from edge_llm_scheduler.core.node_manager import NodeManager
from edge_llm_scheduler.policies.placement import DefaultPlacement, E2Placement


def make_nodes(n: int = 3) -> list:
    return [
        Node(
            node_id=f"node_{i}",
            capability=NodeCapability(compute_flops=100.0 + i * 50, memory_total=16 * 1024**3,
                                      bandwidth=100.0, supported_roles={NodeRole.GENERAL}),
            state=NodeState(memory_free=12 * 1024**3, load=0.05 + 0.05 * i),
        )
        for i in range(n)
    ]


def e2_hash_fn(prompt, block_size):
    """与数据集结构匹配：前缀块 hash 一致，后缀独立。"""
    first = prompt[0]
    split = len(prompt)
    for i, v in enumerate(prompt):
        if v != first:
            split = i
            break
    blocks = []
    for _ in range(max(1, split // block_size)):
        blocks.append(hash((first,)))
    rest = prompt[split:]
    for i in range(0, len(rest), block_size):
        blocks.append(hash(tuple(rest[i:i + block_size])))
    return blocks


async def run_policy(name: str, loader: DatasetLoader, nodes: list, storage: MockKVStore) -> dict:
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    for n in nodes:
        await nm.register(n)

    if name == "e2":
        policy = E2Placement(storage=storage, hash_fn=e2_hash_fn, block_size_tokens=16)
    else:
        policy = DefaultPlacement()
    scheduler = TaskScheduler(node_manager=nm, storage=storage, event_bus=bus,
                              placement_policy=policy)
    from edge_llm_scheduler.backends.mock_engine import MockEngine
    for n in nodes:
        scheduler.attach_engine(n.node_id, MockEngine(n.node_id, tokens_per_sec=99999,
                                                      prefill_ms_per_token=0.0001))

    # 预分布前缀缓存
    prefix_to_node = {}
    for i, pid in enumerate(loader.group_by_prefix().keys()):
        node_id = f"node_{i % len(nodes)}"
        prefix_to_node[pid] = node_id
        prefix_h = abs(hash(pid))
        await storage.save(KVBlock(block_hash=hash((prefix_h,)), num_tokens=16, byte_size=1000),
                           f"{node_id}:gpu")

    total_hit, total_prompt = 0, 0
    node_counts = {}
    for req in loader.requests:
        prefix_h = abs(hash(req["prefix_id"]))
        prompt_len = len(req["input_ids"])
        suffix_len = len(req["suffix_ids"])
        prefix_len = prompt_len - suffix_len
        r = InferenceRequest(request_id=req["request_id"],
                             prompt=[prefix_h] * prefix_len + list(range(suffix_len)),
                             max_tokens=req["max_new_tokens"])
        tasks = await scheduler.route_request(r)
        if not tasks:
            continue
        node_counts[tasks[0].node_id] = node_counts.get(tasks[0].node_id, 0) + 1
        if tasks[0].node_id == prefix_to_node.get(req["prefix_id"]):
            total_hit += prefix_len
        total_prompt += prompt_len

    await bus.stop()
    return {"name": name, "hit_rate": total_hit / max(1, total_prompt),
            "node_counts": node_counts}


@pytest.mark.asyncio
async def test_e2_hit_rate_beats_default():
    loader = DatasetLoader()
    loader.load_preble_workload(num_prefixes=4, reqs_per_prefix=8, cold_ratio=0.2)
    nodes = make_nodes(3)

    default_r = await run_policy("default", loader, nodes, MockKVStore())
    e2_r = await run_policy("e2", loader, nodes, MockKVStore())

    # E2 命中率显著高于 Default（缓存感知路由生效）
    assert e2_r["hit_rate"] > default_r["hit_rate"] + 0.2, \
        f"E2 hit={e2_r['hit_rate']:.3f} should beat default={default_r['hit_rate']:.3f}"


@pytest.mark.asyncio
async def test_e2_load_more_balanced():
    loader = DatasetLoader()
    loader.load_preble_workload(num_prefixes=4, reqs_per_prefix=8, cold_ratio=0.2)
    nodes = make_nodes(3)

    default_r = await run_policy("default", loader, nodes, MockKVStore())
    e2_r = await run_policy("e2", loader, nodes, MockKVStore())

    # E2 分散到多节点（Default 常堆一个）
    assert len(e2_r["node_counts"]) > len(default_r["node_counts"])


@pytest.mark.asyncio
async def test_contextpilot_fixture_loads():
    """真实 fixture 能加载（验证数据集加载器）。"""
    loader = DatasetLoader()
    loader.load_contextpilot_json(
        "moe-infinity/benchmarks/contextpilot/fixtures/shared_prefix_rag.json"
    )
    assert loader.count() == 5
    assert len(loader.group_by_prefix()) == 1
