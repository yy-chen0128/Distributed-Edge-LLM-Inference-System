"""Probe the transformers-5.x DynamicCache API and the true KV bytes.

ASCII only.
"""
import asyncio
import sys

sys.path.insert(0, ".")

import torch
from transformers.cache_utils import DynamicCache

from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine
from edge_llm_scheduler.core.types import StageAssignment

MODEL = ".models/Qwen2.5-0.5B-Instruct"


def dump_api(cache):
    print("  public attrs:", [a for a in dir(cache) if not a.startswith("_")])
    for attr in ("layers", "key_cache", "value_cache"):
        value = getattr(cache, attr, None)
        if value is None:
            print(f"  cache.{attr}: <absent>")
            continue
        try:
            print(f"  cache.{attr}: type={type(value).__name__} len={len(value)}")
        except TypeError:
            print(f"  cache.{attr}: type={type(value).__name__} (no len)")
    layers = getattr(cache, "layers", None)
    if layers:
        layer0 = layers[0]
        print(f"  layers[0]: type={type(layer0).__name__}")
        print(f"  layers[0] attrs:", [a for a in dir(layer0) if not a.startswith("_")])
        for attr in ("keys", "values", "key", "value"):
            t = getattr(layer0, attr, None)
            if t is not None:
                try:
                    print(f"    layers[0].{attr}: shape={tuple(t.shape)} "
                          f"dtype={t.dtype} bytes={t.numel() * t.element_size()}")
                except Exception as exc:
                    print(f"    layers[0].{attr}: {type(exc).__name__}: {exc}")


def total_kv_bytes(cache) -> tuple[int, int]:
    """Return (total_bytes, counted_layers) using whatever API exists."""
    total = 0
    counted = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if t is not None and hasattr(t, "numel"):
                    total += t.numel() * t.element_size()
            counted += 1
        return total, counted
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is not None and values is not None:
        for k, v in zip(keys, values):
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
            counted += 1
    return total, counted


async def main():
    engine = HFLayeredEngine(node_id="kv", model_path=MODEL, device="cuda:0",
                             wire_dtype="float16", logger=lambda *a, **k: None)
    await engine.prepare_epoch(
        StageAssignment(pipeline_epoch=1, node_id="kv", layer_range=(0, 6)))
    await engine.activate_epoch(1)
    cfg = engine.config
    heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    print(f"config: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"kv_heads={heads} head_dim={head_dim} dtype={engine.dtype}")
    per_token_per_layer = 2 * heads * head_dim * torch.tensor([], dtype=engine.dtype).element_size()
    print(f"theoretical KV per token per layer = {per_token_per_layer} bytes")

    rid = "kvprobe"
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    engine.prefill(rid, token_ids=list(range(38)))
    torch.cuda.synchronize()
    delta = torch.cuda.memory_allocated() - base
    cache = engine._caches[(1, rid)]

    print("\n--- DynamicCache API ---")
    dump_api(cache)
    print(f"\n  get_seq_length() = {cache.get_seq_length()}")
    total, counted = total_kv_bytes(cache)
    print(f"  TRUE kv bytes    = {total / 1e6:.3f} MB over {counted} cache slots")
    expect_6 = per_token_per_layer * 38 * 6
    print(f"  expected 6 layers x 38 tok = {expect_6 / 1e6:.3f} MB")
    print(f"  memory_allocated delta     = {delta / 1e6:.3f} MB "
          f"({delta / max(1, total):.1f}x the KV itself)")

    engine.release_request(rid)
    torch.cuda.synchronize()
    print(f"\n  after release: allocated delta = "
          f"{(torch.cuda.memory_allocated() - base) / 1e6:.3f} MB")
    await engine.shutdown()


asyncio.run(main())
