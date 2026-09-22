"""KV 计量回归测试（不需要真实模型即可跑）。

背景：`kv_bytes` 曾经用 `cache[i][:2]` 取值，在 transformers 5.x 上抛
`TypeError: 'DynamicCache' object is not subscriptable`，被 `except ... continue`
吞掉后**恒返回 0**——四机运行里每个 stage 的 `kv_bytes_last` 都是 0 就是这个原因。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine


class FakeLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


def engine_with_cache(cache) -> HFLayeredEngine:
    """构造一个只带 _caches 的引擎实例，避免加载真实模型。"""
    engine = HFLayeredEngine.__new__(HFLayeredEngine)
    engine._caches = {(1, "r1"): cache}
    engine._active_epoch = 1
    return engine


def test_reads_transformers_v5_layers():
    """transformers >= 5：cache.layers[i].keys / .values。"""
    shape = (1, 2, 38, 64)
    cache = SimpleNamespace(layers=[
        FakeLayer(torch.zeros(*shape, dtype=torch.bfloat16),
                  torch.zeros(*shape, dtype=torch.bfloat16)),
        FakeLayer(torch.zeros(*shape, dtype=torch.bfloat16),
                  torch.zeros(*shape, dtype=torch.bfloat16)),
    ])
    per_tensor = 1 * 2 * 38 * 64 * 2          # bf16 = 2 bytes
    assert engine_with_cache(cache).kv_bytes("r1") == 4 * per_tensor


def test_supports_legacy_key_value_cache():
    """transformers 4.4x：cache.key_cache / cache.value_cache（list of tensor）。"""
    shape = (1, 2, 10, 64)
    cache = SimpleNamespace(
        key_cache=[torch.zeros(*shape, dtype=torch.float16)],
        value_cache=[torch.zeros(*shape, dtype=torch.float16)],
    )
    per_tensor = 1 * 2 * 10 * 64 * 2          # fp16 = 2 bytes
    assert engine_with_cache(cache).kv_bytes("r1") == 2 * per_tensor


def test_ignores_empty_slots_and_missing_request():
    """空 slot 贡献 0；未知 request 返回 0。"""
    cache = SimpleNamespace(layers=[
        FakeLayer(None, None),
        FakeLayer(torch.zeros(1, 2, 4, 8, dtype=torch.float32), None),
    ])
    engine = engine_with_cache(cache)
    assert engine.kv_bytes("r1") == 1 * 2 * 4 * 8 * 4   # fp32 = 4 bytes
    assert engine.kv_bytes("nope") == 0


def test_never_reads_by_subscript():
    """回归：旧的 `cache[i]` 写法在 v5 上会抛 TypeError 并被吞掉。"""

    class SubscriptOnly:
        """只有 __getitem__、没有 layers/key_cache 的 cache（v5 的形状）。"""

        def __getitem__(self, item):
            raise TypeError("'DynamicCache' object is not subscriptable")

    engine = engine_with_cache(SubscriptOnly())
    assert engine.kv_bytes("r1") == 0    # 不再抛异常，也不再假装有 KV


def test_request_positions_reports_per_request_progress():
    """进度读数：每个在飞请求"已经处理到第几个 token"。

    这就是中断恢复需要的 progress_tokens 来源——引擎自己就能报，
    不必去读 KV 张量。旧 epoch 的条目不应出现在结果里。
    """

    class FakeCache:
        def __init__(self, seq_len):
            self._seq_len = seq_len

        def get_seq_length(self):
            return self._seq_len

    engine = HFLayeredEngine.__new__(HFLayeredEngine)
    engine._caches = {
        (1, "r1"): FakeCache(38),
        (1, "r2"): FakeCache(10),
        (0, "stale"): FakeCache(5),      # 旧 epoch → 必须被过滤掉
    }
    engine._active_epoch = 1

    assert engine.request_positions() == {"r1": 38, "r2": 10}


def test_request_positions_survives_a_broken_cache_object():
    """某个 cache 读不出长度时不要炸，跳过它即可。"""

    class Broken:
        def get_seq_length(self):
            raise RuntimeError("boom")

    engine = HFLayeredEngine.__new__(HFLayeredEngine)
    engine._caches = {(1, "r1"): Broken()}
    engine._active_epoch = 1
    assert engine.request_positions() == {}
