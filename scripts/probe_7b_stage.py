"""Measure a real 7B stage on this GPU: per-layer cost, so the 0.5B->7B
projection in the docs can be replaced by a measurement.

Only a slice of the model is loaded (a middle stage has no embedding table, the
first stage does), which is exactly what a pipeline stage would hold.

ASCII only.
"""
import asyncio
import statistics
import sys
import time

sys.path.insert(0, ".")

import torch

from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine
from edge_llm_scheduler.core.types import StageAssignment

MODEL = ".models/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0"
P_VALUES = [1, 8, 32, 128, 512]
REPEATS = 5


def fit_line(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    c1 = sxy / sxx if sxx else 0.0
    c0 = my - c1 * mx
    return c0, c1


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def median_ms(fn, repeats=REPEATS, warmup=3):
    for _ in range(warmup):
        fn()
    return statistics.median([timed(fn) for _ in range(repeats)])


async def measure(layer_range, label):
    engine = HFLayeredEngine(node_id="s", model_path=MODEL, device=DEVICE,
                             wire_dtype="float16", logger=lambda *a, **k: None)
    t0 = time.perf_counter()
    await engine.prepare_epoch(
        StageAssignment(pipeline_epoch=1, node_id="s", layer_range=layer_range))
    prepare_ms = (time.perf_counter() - t0) * 1000.0
    await engine.activate_epoch(1)

    layers = layer_range[1] - layer_range[0]
    stage = engine._active_stage()
    param_bytes = stage.param_bytes
    is_first = layer_range[0] == 0
    hidden = engine.hidden_size
    print(f"\n=== 7B stage {label} layers {layer_range[0]}..{layer_range[1]} "
          f"({layers} layers) ===")
    print(f"  prepare_ms={prepare_ms:.0f}  resident={param_bytes / 1e6:.1f} MB "
          f"({param_bytes / 1e6 / layers:.1f} MB/layer)  "
          f"vram_allocated={torch.cuda.memory_allocated() / 1e6:.1f} MB")

    # a middle stage consumes hidden states; synthesise them
    fake = torch.randn(1, 1, hidden, dtype=engine.dtype, device=DEVICE)

    print("  prefill curve (P tokens in one call):")
    xs, ys = [], []
    for p in P_VALUES:
        rid = f"p{p}"
        if is_first:
            fn = (lambda p=p, rid=rid:
                  engine.prefill(rid, token_ids=list(range(p))))
        else:
            h = torch.randn(1, p, hidden, dtype=engine.dtype, device=DEVICE)
            fn = (lambda rid=rid, h=h: engine.prefill(rid, hidden=h))
        engine.release_request(rid)
        med = median_ms(fn)
        xs.append(p)
        ys.append(med)
        print(f"     P={p:>4}  {med:9.2f} ms   {med / p:8.3f} ms/token")
        engine.release_request(rid)
    c0, c1 = fit_line(xs, ys)
    print(f"     FIT  T(P) = {c0:.3f} + {c1:.5f} * P   (ms)")
    print(f"     -> fixed per call = {c0:.3f} ms ({c0 / layers:.3f} ms/layer), "
          f"marginal = {c1:.5f} ms/token")

    rid = "dec"
    engine.release_request(rid)
    if is_first:
        engine.prefill(rid, token_ids=list(range(38)))
    else:
        engine.prefill(rid, hidden=torch.randn(1, 38, hidden, dtype=engine.dtype,
                                               device=DEVICE))
    steps = []
    for _ in range(12):
        h = torch.randn(1, 1, hidden, dtype=engine.dtype, device=DEVICE)
        if is_first:
            steps.append(timed(lambda: engine.decode(rid, token_id=7)))
        else:
            steps.append(timed(lambda h=h: engine.decode(rid, hidden=h)))
    dec = statistics.median(steps[4:])
    print(f"  decode step (1 token): {dec:.3f} ms  "
          f"({dec / layers:.3f} ms/layer)  kv={engine.kv_bytes(rid)} B  "
          f"pos={engine.request_positions()}")
    engine.release_request(rid)

    await engine.shutdown()
    torch.cuda.empty_cache()
    return {"layers": layers, "param_bytes": param_bytes, "prepare_ms": prepare_ms,
            "c0": c0, "c1": c1, "decode": dec}


async def main():
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
          f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    results = []
    # middle stage (no embedding) - the common case in a pipeline
    results.append(await measure((12, 16), "middle"))
    # first stage (carries the 1.1 GB embedding table)
    results.append(await measure((0, 4), "first"))

    print("\n=== summary (7B, measured) ===")
    for r in results:
        print(f"  {r['layers']} layers: resident={r['param_bytes'] / 1e6:7.1f} MB  "
              f"prepare={r['prepare_ms']:6.0f} ms  c0={r['c0']:8.3f} ms "
              f"({r['c0'] / r['layers']:.3f}/layer)  c1={r['c1']:.5f} ms/token  "
              f"decode={r['decode']:8.3f} ms ({r['decode'] / r['layers']:.3f}/layer)")

    print("\n=== compare with the 0.5B measurement ===")
    print("  0.5B: 0.97-0.99 ms/layer fixed, 0.00214 ms/token/layer marginal, "
          "decode ~0.85 ms/layer")
    for r in results:
        print(f"  7B  ({r['layers']}L): {r['c0'] / r['layers']:.3f} ms/layer fixed, "
              f"{r['c1'] / r['layers']:.5f} ms/token/layer marginal, "
              f"{r['decode'] / r['layers']:.3f} ms/layer decode")


asyncio.run(main())
