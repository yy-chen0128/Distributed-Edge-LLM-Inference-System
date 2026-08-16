"""测试：E2Placement 缓存感知路由。

验证：
- exploit：命中长前缀时 → 路由到持有缓存的节点
- explore：命中短/无 → 路由到负载最轻节点（含驱逐代价）
- 命中节点过载时退回 explore
- 无存储时退回纯负载均衡
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core.types import (
    InferenceRequest, KVBlock, Node, NodeCapability, NodeState,
)
from edge_llm_scheduler.backends import MockKVStore
from edge_llm_scheduler.policies.placement import E2Placement


def make_node(nid: str, load: float, mem_free: int = 10 * 1024**3) -> Node:
    return Node(
        node_id=nid,
        capability=NodeCapability(compute_flops=100.0, memory_total=16 * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=mem_free, load=load),
    )


def make_request(prompt_len: int = 64, rid: str = "r1") -> InferenceRequest:
    return InferenceRequest(request_id=rid, prompt="a" * prompt_len, max_tokens=16)


def make_policy(storage, **kw) -> E2Placement:
    """构造 E2Placement，注入确定性 hash：块 i 的 hash = 100+i。"""
    def hash_fn(prompt, block_size):
        n = len(prompt) // block_size + (1 if len(prompt) % block_size else 0)
        return [100 + i for i in range(n)]
    return E2Placement(storage=storage, hash_fn=hash_fn, **kw)


@pytest.mark.asyncio
async def test_exploit_routes_to_hit_holder():
    """长命中前缀 → 路由到持有缓存的节点（即使它负载更高）。"""
    storage = MockKVStore()
    # node_a 持有前 3 块缓存（块 hash = 100,101,102；64 token → 4 块）
    for h in [100, 101, 102]:
        await storage.save(KVBlock(block_hash=h, num_tokens=16, byte_size=1000,
                                   reuse_count=5, prefill_time_ms=10.0), "node_a:gpu")

    nodes = [make_node("node_a", load=0.8), make_node("node_b", load=0.1)]
    req = make_request(prompt_len=64)

    policy = make_policy(storage)
    tasks = await policy.place(model=None, nodes=nodes, request=req, storage=storage)

    # 命中 3 块(48 token) > 剩余 1 块(16 token) → exploit → node_a
    assert tasks and tasks[0].node_id == "node_a"


@pytest.mark.asyncio
async def test_explore_when_hit_short():
    """命中短（explore 更划算）→ 路由到负载最轻。"""
    storage = MockKVStore()
    # node_a 只持有 1 块缓存（16 token），但负载高
    await storage.save(KVBlock(block_hash=100, num_tokens=16, byte_size=1000), "node_a:gpu")

    nodes = [make_node("node_a", load=0.8), make_node("node_b", load=0.1)]
    req = make_request(prompt_len=64)

    policy = make_policy(storage)
    tasks = await policy.place(model=None, nodes=nodes, request=req, storage=storage)

    # 命中 1 块(16 token) < 剩余 3 块(48 token) → explore → node_b
    assert tasks and tasks[0].node_id == "node_b"


@pytest.mark.asyncio
async def test_explore_considers_eviction_cost():
    """explore 时，持有低价值 KV 的节点更受欢迎（驱逐它代价小）。"""
    storage = MockKVStore()
    # node_a 持有高价值块（reuse_count 高 → 驱逐贵）
    await storage.save(KVBlock(block_hash=1, num_tokens=16, byte_size=1000,
                               reuse_count=100, prefill_time_ms=50.0), "node_a:gpu")
    # node_b 持有低价值块（驱逐便宜）
    await storage.save(KVBlock(block_hash=2, num_tokens=16, byte_size=1000,
                               reuse_count=1, prefill_time_ms=1.0), "node_b:gpu")

    # 两者负载相同 → 驱逐代价决定：node_b 驱逐便宜 → 选它
    nodes = [make_node("node_a", load=0.3), make_node("node_b", load=0.3)]
    req = make_request(prompt_len=64)

    policy = make_policy(storage)
    tasks = await policy.place(model=None, nodes=nodes, request=req, storage=storage)

    assert tasks and tasks[0].node_id == "node_b"


@pytest.mark.asyncio
async def test_exploit_falls_back_when_hit_node_overloaded():
    """命中节点过载 → 退回 explore。"""
    storage = MockKVStore()
    for h in [100, 101, 102]:
        await storage.save(KVBlock(block_hash=h, num_tokens=16, byte_size=1000), "node_a:gpu")

    # node_a 持有缓存但负载爆满（load=1.0，超过阈值 0.95）
    nodes = [make_node("node_a", load=1.0), make_node("node_b", load=0.2)]
    req = make_request(prompt_len=64)

    policy = make_policy(storage)
    tasks = await policy.place(model=None, nodes=nodes, request=req, storage=storage)

    assert tasks and tasks[0].node_id == "node_b"


@pytest.mark.asyncio
async def test_no_storage_falls_back_to_load_balancing():
    """无存储（None）→ 纯负载均衡。"""
    nodes = [make_node("node_a", load=0.9), make_node("node_b", load=0.1)]
    req = make_request(prompt_len=64)

    policy = make_policy(None)
    tasks = await policy.place(model=None, nodes=nodes, request=req, storage=None)

    assert tasks and tasks[0].node_id == "node_b"
