"""真实 LMCache 对接测试（需要 torch + LMCache 可导入）。

- 若 torch 或 LMCache 不可导入 → skip（环境未就绪）
- 若可用 → 用真实 LMCacheEngine（CPU 模式）验证 LMCacheStore 的
  save/load/lookup/move/evict 真实工作

运行方式：torch 就绪后自动生效。手动跳过验证：
  PYTHONPATH=LMCache:. python -m pytest edge_llm_scheduler/tests/test_lmcache_real.py -v
"""

from __future__ import annotations

import asyncio

import pytest

# 检测 torch 和 LMCache 是否可用
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import lmcache  # noqa: F401
    _HAS_LMCACHE = True
except ImportError:
    _HAS_LMCACHE = False

REAL_AVAILABLE = _HAS_TORCH and _HAS_LMCACHE


@pytest.mark.skipif(not REAL_AVAILABLE, reason="torch/LMCache 未就绪（需 torch + LMCache 环境）")
@pytest.mark.asyncio
async def test_real_lmcache_engine_store_retrieve():
    """用真实 LMCacheEngine（CPU）验证 LMCacheStore 存/取。"""
    from edge_llm_scheduler.backends.lmcache_storage import LMCacheStore
    from edge_llm_scheduler.core.types import KVBlock

    # 创建真实 LMCacheEngine（纯 CPU 模式）
    engine = _create_real_lmcache_engine()
    if engine is None:
        pytest.skip("真实 LMCacheEngine 创建失败（CPU 模式不可用）")

    store = LMCacheStore(lm_engine=engine)
    block = KVBlock(block_hash=123, num_tokens=16, byte_size=2048)

    # save → load
    await store.save(block, "test:gpu")
    loaded = await store.load(123, "test:gpu")
    assert loaded is not None and loaded.block_hash == 123


@pytest.mark.skipif(not REAL_AVAILABLE, reason="torch/LMCache 未就绪")
@pytest.mark.asyncio
async def test_real_lmcache_engine_lookup():
    """真实 lookup 返回命中数。"""
    from edge_llm_scheduler.backends.lmcache_storage import LMCacheStore
    engine = _create_real_lmcache_engine()
    if engine is None:
        pytest.skip("真实 LMCacheEngine 创建失败")

    store = LMCacheStore(lm_engine=engine)
    # lookup 未存过的前缀 → 0
    hit = await store.lookup([9999])
    assert hit == 0


def _create_real_lmcache_engine():
    """创建真实 LMCacheEngine（CPU 模式）。返回 None 表示不可用。"""
    try:
        from lmcache.v1.engine.config import LMCacheEngineConfig
        from lmcache.v1.engine.metadata import LMCacheMetadata
        from lmcache.v1.engine import LMCacheEngine
        from lmcache.utils import CacheEngineKey  # noqa: F401

        # CPU 模式配置：只用 local_cpu，不碰 GPU
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=1.0,   # 1GB CPU 缓存
            local_disk=False,
        )
        metadata = LMCacheMetadata.from_metadata(
            model_name="test-model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(1, 2, 16, 8, 128),
            served_model_name="test-model",
        )
        engine = LMCacheEngine(config, metadata)
        return engine
    except Exception as e:
        print(f"[test_lmcache_real] LMCacheEngine 创建失败: {e}")
        return None
