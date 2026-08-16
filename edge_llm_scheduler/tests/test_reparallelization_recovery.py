"""测试：CapabilityReparallelization（按能力重切层） + TokenRecovery（token 级恢复）。

验证：
- 重并行化：算力强的节点分更多层，且层区间连续覆盖全部模型
- 中断恢复：保留 progress_tokens，从该位置继续（而非从头）
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.core.types import ModelSpec, Node, NodeCapability, NodeState, Task
from edge_llm_scheduler.policies.reparallelization import CapabilityReparallelization
from edge_llm_scheduler.policies.recovery import TokenRecovery


def make_node(nid: str, flops: float) -> Node:
    return Node(
        node_id=nid,
        capability=NodeCapability(compute_flops=flops, memory_total=16 * 1024**3, bandwidth=100.0),
        state=NodeState(memory_free=12 * 1024**3, load=0.0),
    )


def make_model(layers: int = 16) -> ModelSpec:
    return ModelSpec(model_name="m", num_layers=layers, hidden_size=1024)


@pytest.mark.asyncio
async def test_reparallelize_strong_node_gets_more_layers():
    """算力 2:1 → 强节点分约 2 倍层。"""
    model = make_model(layers=12)
    nodes = [make_node("strong", flops=200.0), make_node("weak", flops=100.0)]
    policy = CapabilityReparallelization()
    placements = await policy.reconfigure(model, nodes, current_placements={})

    by_id = {p.node_id: p for p in placements}
    strong = by_id["strong"].layer_range
    weak = by_id["weak"].layer_range
    assert (strong[1] - strong[0]) == 2 * (weak[1] - weak[0])  # 8:4
    # 层区间连续覆盖全部 12 层
    assert strong[0] == 0 and weak[1] == 12
    assert strong[1] == weak[0]


@pytest.mark.asyncio
async def test_reparallelize_covers_all_layers():
    """多个节点 → 层区间连续覆盖全部。"""
    model = make_model(layers=16)
    nodes = [make_node("a", flops=100.0), make_node("b", flops=100.0), make_node("c", flops=100.0)]
    policy = CapabilityReparallelization()
    placements = await policy.reconfigure(model, nodes, current_placements={})

    ranges = sorted(p.layer_range for p in placements)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 16
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]  # 连续


@pytest.mark.asyncio
async def test_reparallelize_skips_dead_nodes():
    """死节点不参与分配。"""
    model = make_model(layers=10)
    dead = make_node("dead", flops=100.0)
    dead.state.alive = False
    alive = make_node("alive", flops=100.0)
    policy = CapabilityReparallelization()
    placements = await policy.reconfigure(model, [dead, alive], current_placements={})
    assert len(placements) == 1
    assert placements[0].node_id == "alive"
    assert placements[0].layer_range == (0, 10)


@pytest.mark.asyncio
async def test_no_model_returns_current():
    """model 为 None → 返回当前放置。"""
    policy = CapabilityReparallelization()
    placements = await policy.reconfigure(None, [make_node("a", 100.0)], current_placements={"a": None})
    assert placements == [None]


@pytest.mark.asyncio
async def test_token_recovery_resumes_from_progress():
    """token 级恢复：保留 progress_tokens 和已提交 KV。"""
    task = Task(task_id="t1", request_id="r1", node_id="n1", kv_block_ids=[10, 11],
                progress_tokens=32)
    policy = TokenRecovery()
    retry = await policy.recover(task)

    assert retry is not None
    assert retry.progress_tokens == 32  # 从这继续
    assert retry.kv_block_ids == [10, 11]  # 已提交 KV 保留
    assert retry.request_id == "r1"
    assert retry._retries == 1


@pytest.mark.asyncio
async def test_token_recovery_gives_up_after_max():
    task = Task(task_id="t1", request_id="r1", node_id="n1")
    task._retries = 3
    policy = TokenRecovery(max_retries=3)
    assert await policy.recover(task) is None
