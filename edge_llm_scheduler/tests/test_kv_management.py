"""验证：KV 存储抽象（save/load/lookup/move/evict/get_index）。"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.backends import MockKVStore
from edge_llm_scheduler.core.types import KVBlock


def block(h: int, tokens: int = 16, hot: bool = False) -> KVBlock:
    return KVBlock(
        block_hash=h, num_tokens=tokens, byte_size=tokens * 128,
        reuse_count=10 if hot else 0, prefill_time_ms=5.0, is_hot=hot,
    )


@pytest.mark.asyncio
async def test_save_load():
    s = MockKVStore()
    b = block(1)
    await s.save(b, "node_a:gpu")
    got = await s.load(1, "node_a:gpu")
    assert got is not None and got.block_hash == 1
    assert await s.load(1, "node_b:gpu") is None


@pytest.mark.asyncio
async def test_lookup_continuous_hit():
    s = MockKVStore()
    await s.save(block(1), "a:gpu")
    await s.save(block(2), "a:gpu")
    await s.save(block(3), "a:gpu")

    # 连续命中 3 块
    assert await s.lookup([1, 2, 3]) == 3
    # 中断：3 后面接 99 不命中
    assert await s.lookup([1, 2, 99]) == 2
    # 从头就不同
    assert await s.lookup([99, 1, 2]) == 0
    # 限定位置
    await s.save(block(4), "b:gpu")
    assert await s.lookup([4], locations=["b:gpu"]) == 1
    assert await s.lookup([4], locations=["a:gpu"]) == 0


@pytest.mark.asyncio
async def test_move_and_evict():
    s = MockKVStore()
    await s.save(block(1), "a:gpu")
    await s.move(1, "a:gpu", "b:cpu")
    assert await s.load(1, "b:cpu") is not None

    index = await s.get_index()
    assert 1 in index and "b:cpu" in index[1]

    await s.evict(1, "b:cpu")
    assert await s.load(1, "b:cpu") is None
    assert 1 not in await s.get_index()


@pytest.mark.asyncio
async def test_index_tracks_locations():
    s = MockKVStore()
    await s.save(block(1), "a:gpu")
    await s.save(block(1), "b:cpu")  # 同一块多副本
    index = await s.get_index()
    assert set(index[1]) == {"a:gpu", "b:cpu"}


@pytest.mark.asyncio
async def test_block_priority():
    hot = block(1, hot=True)
    cold = block(2)
    # hot 块 reuse_count 更高 → priority 更高
    assert hot.priority > cold.priority
