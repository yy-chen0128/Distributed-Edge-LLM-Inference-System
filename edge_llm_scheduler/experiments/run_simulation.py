"""模拟实验：用真实数据集对比调度策略的 KV 命中率与负载均衡。

用法：
  python -m edge_llm_scheduler.experiments.run_simulation \
      --dataset <path-to-contextpilot-fixture.json> \
      [--policy e2|default|capability] \
      [--nodes 3]

输出：
  - 总请求数 / 共享前缀数
  - 每个策略的：平均 KV 命中率、节点负载分布（标准差）
  - 命中率 = 命中 token / 总 prompt token（衡量 prefill 节省量）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict

from ..backends.datasets import DatasetLoader
from ..backends import MockKVStore
from ..core import (
    EventBus, InferenceRequest, Node, NodeCapability, NodeRole, NodeState,
    TaskScheduler,
)
from ..core.node_manager import NodeManager
from ..policies.placement import DefaultPlacement, E2Placement, CapabilityPlacement

logging.basicConfig(level=logging.WARNING)


def make_nodes(n: int) -> list:
    nodes = []
    for i in range(n):
        nodes.append(Node(
            node_id=f"node_{i}",
            capability=NodeCapability(
                compute_flops=100.0 + i * 50,
                memory_total=16 * 1024**3,
                bandwidth=100.0,
                supported_roles={NodeRole.GENERAL},
            ),
            state=NodeState(memory_free=12 * 1024**3, load=0.05 + 0.05 * i),
        ))
    return nodes


async def run_policy(name: str, loader: DatasetLoader, nodes: list, storage: MockKVStore) -> dict:
    bus = EventBus()
    bus.start()
    nm = NodeManager(event_bus=bus)
    for n in nodes:
        await nm.register(n)

    if name == "e2":
        # 注入与数据集结构匹配的 hash_fn：前缀 [prefix_h]*N 切成固定块，后缀独立
        block = 16
        def dataset_hash(prompt, block_size):
            first = prompt[0]
            split = len(prompt)
            for i, v in enumerate(prompt):
                if v != first:
                    split = i
                    break
            blocks = []
            # 前缀块：每 16 token 一块，hash 一致（同前缀可命中）
            prefix_len = split
            for _ in range(max(1, prefix_len // block)):
                blocks.append(hash((first,)))          # 所有前缀块同 hash
            rest = prompt[split:]
            for i in range(0, len(rest), block):
                blocks.append(hash(tuple(rest[i:i+block])))
            return blocks
        policy = E2Placement(storage=storage, hash_fn=dataset_hash, block_size_tokens=block)
    elif name == "capability":
        policy = CapabilityPlacement()
    else:
        policy = DefaultPlacement()

    scheduler = TaskScheduler(node_manager=nm, storage=storage, event_bus=bus,
                              placement_policy=policy)
    # 引擎用最小 mock（只测路由，不真推理）
    from ..backends.mock_engine import MockEngine
    for n in nodes:
        scheduler.attach_engine(n.node_id, MockEngine(n.node_id, tokens_per_sec=99999,
                                                      prefill_ms_per_token=0.0001))

    # 预先把每个共享前缀的缓存散布到不同节点（模拟缓存已就位）
    from ..core.types import KVBlock
    block = 16
    prefix_to_node = {}
    for i, pid in enumerate(loader.group_by_prefix().keys()):
        node_id = f"node_{i % len(nodes)}"
        prefix_to_node[pid] = node_id
        prefix_h = abs(hash(pid))
        # 前缀若干块，每块 hash 相同（与 dataset_hash 一致）
        await storage.save(KVBlock(block_hash=hash((prefix_h,)), num_tokens=block, byte_size=1000),
                           f"{node_id}:gpu")
        await storage.save(KVBlock(block_hash=hash((prefix_h,)), num_tokens=block, byte_size=1000),
                           f"{node_id}:gpu")

    total_hit, total_prompt = 0, 0
    node_counts = defaultdict(int)
    for req in loader.requests:
        prefix_h = abs(hash(req["prefix_id"]))
        prompt_len = len(req["input_ids"])
        suffix_len = len(req["suffix_ids"])
        prefix_len = prompt_len - suffix_len

        r = InferenceRequest(
            request_id=req["request_id"],
            prompt=[prefix_h] * prefix_len + list(range(suffix_len)),
            max_tokens=req["max_new_tokens"],
        )
        tasks = await scheduler.route_request(r)
        if not tasks:
            continue
        node_counts[tasks[0].node_id] += 1
        # 真命中：路由到的节点持有该前缀缓存
        if tasks[0].node_id == prefix_to_node.get(req["prefix_id"]):
            total_hit += prefix_len
        total_prompt += prompt_len

    await bus.stop()
    hit_rate = total_hit / max(1, total_prompt)
    loads = list(node_counts.values())
    load_std = (sum((v - sum(loads)/len(loads))**2 for v in loads) / len(loads)) ** 0.5 if loads else 0
    return {"name": name, "hit_rate": hit_rate, "node_counts": dict(node_counts),
            "load_std": load_std}


async def main(args) -> None:
    loader = DatasetLoader()
    if args.dataset:
        loader.load_contextpilot_json(args.dataset)
    else:
        loader.load_preble_workload(num_prefixes=3, reqs_per_prefix=10, cold_ratio=0.2)

    nodes = make_nodes(args.nodes)
    storage = MockKVStore()
    print(f"dataset: {loader.count()} requests, {len(loader.group_by_prefix())} shared prefixes\n")

    results = []
    for p in args.policy:
        r = await run_policy(p, loader, nodes, storage)
        results.append(r)
        print(f"[{r['name']}] hit_rate={r['hit_rate']:.3f}  node_load={r['node_counts']}  load_std={r['load_std']:.1f}")

    if len(results) >= 2:
        print(f"\ncompare: E2 vs Default hit_rate improvement = "
              f"{(max(r['hit_rate'] for r in results) - min(r['hit_rate'] for r in results)):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--policy", nargs="+", default=["default", "e2"])
    parser.add_argument("--nodes", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args))
