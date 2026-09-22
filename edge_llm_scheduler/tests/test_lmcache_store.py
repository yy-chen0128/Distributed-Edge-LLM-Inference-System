"""契约测试：LMCacheStore 对接 LMCacheEngine。

不需要 GPU——用一个 fake LMCacheEngine（模拟真实接口签名），
验证 LMCacheStore 调用了正确的方法、传了正确的参数、正确处理返回值。

真实 LMCacheEngine 接口（探索确认）：
  store(tokens=None, hashes=None, offsets=None, mask=None, **kwargs)
  retrieve(tokens, mask=None, **kwargs) -> torch.Tensor  (bool 掩码)
  lookup(tokens=None, hashes=None, offsets=None, search_range=None, ...) -> int
  move(tokens, old_position, new_position, event_id, do_copy=True) -> int
  clear(tokens=None, locations=None, request_configs=None) -> int

本测试用 fake 验证 LMCacheStore 的适配逻辑（方法调用 + 参数 + 返回值处理）。
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from edge_llm_scheduler.backends.lmcache_storage import LMCacheStore
from edge_llm_scheduler.core.types import KVBlock


class FakeLMCacheEngine:
    """模拟 LMCacheEngine 的关键方法，记录调用。"""

    def __init__(self):
        self.calls = []
        self.stored = {}          # tokens -> 存在
        self.lookup_result = 0
        self.retrieve_hit = False

    def store(self, tokens=None, hashes=None, offsets=None, mask=None, **kwargs):
        self.calls.append(("store", tokens, kwargs))
        self.stored[tuple(tokens)] = True

    def retrieve(self, tokens=None, mask=None, **kwargs):
        self.calls.append(("retrieve", tokens, kwargs))
        # 真实返回 torch.Tensor 布尔掩码；fake 返回可迭代的 bool 容器
        class _Mask:
            def __init__(self, hit):
                self._hit = hit
            def __len__(self):
                return 1
            def __getitem__(self, i):
                return self._hit
        return _Mask(self.retrieve_hit)

    def lookup(self, tokens=None, hashes=None, offsets=None, search_range=None, lookup_id=None, pin=False, request_configs=None):
        self.calls.append(("lookup", tokens, {"search_range": search_range, "pin": pin}))
        return self.lookup_result

    def move(self, tokens=None, old_position=None, new_position=None, event_id=None, do_copy=True):
        self.calls.append(("move", tokens, {"old_position": old_position, "new_position": new_position, "event_id": event_id}))
        return 1

    def clear(self, tokens=None, locations=None, request_configs=None):
        self.calls.append(("clear", tokens, {"locations": locations}))
        return 1


@pytest.mark.asyncio
async def test_save_calls_engine_store():
    fake = FakeLMCacheEngine()
    store = LMCacheStore(lm_engine=fake)
    block = KVBlock(block_hash=42, num_tokens=16, byte_size=2048)
    await store.save(block, "node_a:gpu")
    assert fake.calls, "no call made"
    assert fake.calls[0][0] == "store"
    tokens = fake.calls[0][1]
    assert tokens is not None and len(tokens) == 16


@pytest.mark.asyncio
async def test_load_hit_and_miss():
    fake = FakeLMCacheEngine()
    store = LMCacheStore(lm_engine=fake)

    fake.retrieve_hit = True
    block = await store.load(42, "node_a:gpu")
    assert block is not None and block.block_hash == 42
    assert fake.calls[0][0] == "retrieve"

    fake.retrieve_hit = False
    block = await store.load(99, "node_a:gpu")
    assert block is None


@pytest.mark.asyncio
async def test_lookup_returns_hit_count():
    fake = FakeLMCacheEngine()
    fake.lookup_result = 7
    store = LMCacheStore(lm_engine=fake)
    hit = await store.lookup([1, 2, 3, 4])
    assert hit == 7
    assert fake.calls[0][0] == "lookup"


@pytest.mark.asyncio
async def test_move_calls_engine_move():
    fake = FakeLMCacheEngine()
    store = LMCacheStore(lm_engine=fake)
    await store.move(42, "node_a:gpu", "node_b:gpu")
    assert fake.calls[0][0] == "move"
    call = fake.calls[0]
    assert call[1] == [42]
    kwargs = call[2]
    assert kwargs["old_position"] == "node_a:gpu"
    assert kwargs["new_position"][0] == "node_b:gpu"
    assert kwargs["event_id"] == "scheduler"


@pytest.mark.asyncio
async def test_evict_calls_clear():
    fake = FakeLMCacheEngine()
    store = LMCacheStore(lm_engine=fake)
    await store.evict(42, "node_a:gpu")
    assert fake.calls[0][0] == "clear"


@pytest.mark.asyncio
async def test_estimate_move_cost():
    fake = FakeLMCacheEngine()
    store = LMCacheStore(lm_engine=fake)
    store._bandwidth_mbps = 100.0
    block = KVBlock(block_hash=1, num_tokens=16, byte_size=10_000)
    cost = await store.estimate_move_cost(block, "a", "b")
    assert cost > 0


@pytest.mark.asyncio
async def test_catalog_tracks_locations_and_moves():
    fake = FakeLMCacheEngine()
    store = LMCacheStore(lm_engine=fake)
    block = KVBlock(
        block_hash=77,
        num_tokens=4,
        byte_size=512,
        tokens=[10, 11, 12, 13],
    )

    await store.save(block, "node_a:cpu")
    assert await store.get_index() == {77: ["node_a:cpu"]}

    await store.move(77, "node_a:cpu", "node_b:cpu")
    assert await store.get_index() == {77: ["node_b:cpu"]}

    await store.evict(77, "node_b:cpu")
    assert await store.get_index() == {}


@pytest.mark.asyncio
async def test_detached_raises():
    """未 attach LMCacheEngine 时调用应明确报错。"""
    store = LMCacheStore(lm_engine=None)
    with pytest.raises(RuntimeError):
        await store.save(KVBlock(1, 16, 10), "a:gpu")
