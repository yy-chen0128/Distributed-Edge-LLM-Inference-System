"""测试：PriorityMigration PT×N 优先级迁移。

验证：
- 按价值（reuse_count × prefill_time_ms）降序排列迁移
- 时间受限时只传高价值块，低价值块被丢
- 有可用节点时目标选择正确
- 执行后块实际迁移/丢弃
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core.types import KVBlock, Node, NodeCapability, NodeState
from edge_llm_scheduler.backends import MockKVStore
from edge_llm_scheduler.policies.migration import PriorityMigration


def make_node(nid: str, load: float = 0.1, mem_free: int = 10 * 1024**3) -> Node:
    return Node(
        node_id=nid,
        capability=NodeCapability(compute_flops=100.0, memory_total=16 * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=mem_free, load=load),
    )


def make_block(h: int, reuse: int, prefill_ms: float, size: int = 1000) -> KVBlock:
    return KVBlock(block_hash=h, num_tokens=16, byte_size=size,
                   reuse_count=reuse, prefill_time_ms=prefill_ms)


@pytest.mark.asyncio
async def test_sorts_by_priority_desc():
    """迁移计划按价值降序排列。"""
    storage = MockKVStore()
    storage.bandwidth_mbps = 1.0  # 很慢 → 受限
    # 三块：价值分别是 100*50=5000, 10*10=100, 1*1=1
    await storage.save(make_block(1, reuse=100, prefill_ms=50.0), "leave:gpu")
    await storage.save(make_block(2, reuse=10, prefill_ms=10.0), "leave:gpu")
    await storage.save(make_block(3, reuse=1, prefill_ms=1.0), "leave:gpu")

    nodes = [make_node("survivor")]
    policy = PriorityMigration(deadline_ms=1000000, bandwidth_mbps=1.0)
    plan = await policy.decide("leave", [1, 2, 3], storage, available_nodes=nodes)

    priorities = [mv["priority"] for mv in plan.block_moves]
    assert priorities == sorted(priorities, reverse=True)  # 降序
    assert len(plan.block_moves) == 3


@pytest.mark.asyncio
async def test_budget_drops_low_priority_blocks():
    """时间受限：只传高价值块，低价值被丢。"""
    storage = MockKVStore()
    # 每块 cost = 1000B / 100MB/s × 0.001 = 0.01ms
    await storage.save(make_block(1, reuse=100, prefill_ms=50.0), "leave:gpu")  # 价值 5000
    await storage.save(make_block(2, reuse=100, prefill_ms=50.0), "leave:gpu")  # 5000
    await storage.save(make_block(3, reuse=1, prefill_ms=1.0), "leave:gpu")     # 1（低价值）
    await storage.save(make_block(4, reuse=1, prefill_ms=1.0), "leave:gpu")     # 1（低价值）

    nodes = [make_node("survivor")]
    # deadline=0.01ms → 预算只够传 1 块（0.01ms）
    policy = PriorityMigration(deadline_ms=0.01)
    plan = await policy.decide("leave", [1, 2, 3, 4], storage, available_nodes=nodes)

    # 只传价值最高的 1 块，其余 3 块丢
    assert len(plan.block_moves) == 1
    assert len(plan.block_drops) == 3
    assert plan.block_moves[0]["block_hash"] == 1  # 价值最高


@pytest.mark.asyncio
async def test_no_available_node_drops_all():
    """无可用节点 → 全部丢弃。"""
    storage = MockKVStore()
    await storage.save(make_block(1, reuse=100, prefill_ms=50.0), "leave:gpu")

    policy = PriorityMigration()
    plan = await policy.decide("leave", [1], storage, available_nodes=[])

    assert plan.block_moves == []
    assert 1 in plan.block_drops


@pytest.mark.asyncio
async def test_execute_migrates_blocks():
    """执行：块从离开节点迁到目标节点。"""
    storage = MockKVStore()
    await storage.save(make_block(1, reuse=100, prefill_ms=50.0), "leave:gpu")
    await storage.save(make_block(2, reuse=100, prefill_ms=50.0), "leave:gpu")

    nodes = [make_node("survivor")]
    policy = PriorityMigration()   # deadline=inf → 全传
    plan = await policy.decide("leave", [1, 2], storage, available_nodes=nodes)
    await policy.execute(plan, storage=storage)

    index = await storage.get_index()
    # 两块都应迁到 survivor:gpu
    for bid in [1, 2]:
        locs = index.get(bid, [])
        assert "survivor:gpu" in locs, f"block {bid} not migrated"


@pytest.mark.asyncio
async def test_default_migration_matches_interface():
    """DefaultMigration 也接受新接口（向后兼容）。"""
    from edge_llm_scheduler.policies.migration import DefaultMigration
    storage = MockKVStore()
    await storage.save(make_block(1, reuse=5, prefill_ms=10.0), "leave:gpu")
    nodes = [make_node("survivor")]
    policy = DefaultMigration()
    plan = await policy.decide("leave", [1], storage, available_nodes=nodes)
    assert len(plan.block_moves) == 1
