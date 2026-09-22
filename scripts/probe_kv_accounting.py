"""Probe KV-cache accounting in HFLayeredEngine on a real GPU.

Question: the 4-stage run reported kv_bytes_last = 0 at every stage, even though
generation clearly works. Is the KV really there and the accounting broken, or is
no KV being kept at all?

ASCII only (travels through the PowerShell pipe).
"""
import asyncio
import sys

sys.path.insert(0, ".")

import torch

from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine
from edge_llm_scheduler.core.types import StageAssignment

MODEL = ".models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"


async def main():
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    engine = HFLayeredEngine(node_id="kv", model_path=MODEL, device=DEVICE,
                             wire_dtype="float16", logger=lambda *a, **k: None)
    await engine.prepare_epoch(
        StageAssignment(pipeline_epoch=1, node_id="kv", layer_range=(0, 6))
    )
    await engine.activate_epoch(1)

    rid = "kvprobe"
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    print("\n--- prefill 38 tokens ---")
    engine.prefill(rid, token_ids=list(range(38)))
    torch.cuda.synchronize()
    after_prefill = torch.cuda.memory_allocated()
    cache = engine._caches.get((1, rid))
    print(f"  cache object      : {type(cache).__name__}")
    print(f"  len(cache)        : {len(cache) if cache is not None else None}")
    if cache is not None:
        try:
            print(f"  get_seq_length()  : {cache.get_seq_length()}")
        except Exception as exc:
            print(f"  get_seq_length()  : {type(exc).__name__}: {exc}")
        try:
            layer0 = cache[0]
            print(f"  type(cache[0])    : {type(layer0).__name__}")
            print(f"  repr(cache[0])[:120]: {repr(layer0)[:120]}")
            try:
                k, v = layer0[:2]
                print(f"  cache[0][:2]      : key={type(k).__name__} "
                      f"{tuple(k.shape) if hasattr(k, 'shape') else k}, "
                      f"value={type(v).__name__} "
                      f"{tuple(v.shape) if hasattr(v, 'shape') else v}")
            except Exception as exc:
                print(f"  cache[0][:2]      : {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"  cache[0]          : {type(exc).__name__}: {exc}")

    print(f"\n  torch.cuda.memory_allocated delta = "
          f"{(after_prefill - base) / 1e6:.2f} MB")
    print(f"  engine.kv_bytes(rid)              = {engine.kv_bytes(rid)} bytes")
    print(f"  engine.request_positions()        = {engine.request_positions()}")

    print("\n--- 5 decode steps ---")
    for i in range(5):
        engine.decode(rid, token_id=7)
    torch.cuda.synchronize()
    after_decode = torch.cuda.memory_allocated()
    print(f"  torch.cuda.memory_allocated delta = "
          f"{(after_decode - base) / 1e6:.2f} MB")
    print(f"  engine.kv_bytes(rid)              = {engine.kv_bytes(rid)} bytes")
    print(f"  engine.request_positions()        = {engine.request_positions()}")

    print("\n--- second request (position is per-request) ---")
    engine.prefill("second", token_ids=list(range(10)))
    print(f"  positions                         = {engine.request_positions()}")
    print(f"  status()['kv_bytes']              = {engine.status()['kv_bytes']}")

    # Expected size: 2 (K,V) * layers * num_tokens * kv_heads * head_dim * bytes
    cfg = engine.config
    layers = 6
    heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    for tokens in (38, 43):
        expect = 2 * layers * tokens * heads * head_dim * 2
        print(f"  expected KV for {tokens} tokens = {expect / 1e6:.2f} MB "
              f"(2*{layers}L*{tokens}t*{heads}h*{head_dim}d*2B)")

    print("\n--- release_request ---")
    engine.release_request(rid)
    torch.cuda.synchronize()
    print(f"  torch.cuda.memory_allocated delta = "
          f"{(torch.cuda.memory_allocated() - base) / 1e6:.2f} MB")
    print(f"  kv_bytes after release            = {engine.kv_bytes(rid)} bytes")
    print(f"  cache in _caches?                 = {(1, rid) in engine._caches}")
    await engine.shutdown()


asyncio.run(main())
