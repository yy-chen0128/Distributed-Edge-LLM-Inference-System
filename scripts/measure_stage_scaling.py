"""Measure per-call fixed overhead vs per-token marginal cost on a real GPU stage.

Separates T(call) = c0 + c1 * P (P = tokens in the call) so we can say how much
of the wall time is framework overhead rather than model work.

Output labels are ASCII on purpose (they travel through a PowerShell pipe).
"""
import asyncio
import statistics
import sys
import time

sys.path.insert(0, ".")

import torch

from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine
from edge_llm_scheduler.core.types import StageAssignment

MODEL = ".models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"
PROMPT_TOKENS = 38          # matches the recorded 4-stage run
P_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DECODE_STEPS = 20
REPEATS = 7


def fit_line(xs, ys):
    """Least squares y = c0 + c1*x. Returns (c0, c1, r2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    c1 = sxy / sxx if sxx else 0.0
    c0 = my - c1 * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (c0 + c1 * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return c0, c1, r2


def median_ms(fn, repeats=REPEATS, warmup=5):
    """Median wall time with CUDA synchronisation (async kernels must be fenced)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples)


async def build(node_id, layer_range):
    engine = HFLayeredEngine(
        node_id=node_id, model_path=MODEL, device=DEVICE,
        wire_dtype="float16", logger=lambda *a, **k: None,
    )
    await engine.prepare_epoch(
        StageAssignment(pipeline_epoch=1, node_id=node_id, layer_range=layer_range)
    )
    await engine.activate_epoch(1)
    return engine


def stage_param_bytes(engine):
    stage = engine._active_stage()
    return stage.param_bytes


async def measure_stage(layer_range):
    label = f"layers {layer_range[0]}..{layer_range[1]} of {24}"
    engine = await build("probe", layer_range)
    param_bytes = stage_param_bytes(engine)
    layers = layer_range[1] - layer_range[0]
    print(f"\n=== stage {label}  |  resident weights {param_bytes / 1e6:.1f} MB "
          f"({param_bytes / 1e6 / layers:.1f} MB/layer)  |  torch {torch.__version__} "
          f"|  device {torch.cuda.get_device_name(0)} ===")

    # ---- prefill: T(P) for one call carrying P tokens
    print("prefill curve (one call, P tokens):")
    xs, ys = [], []
    counter = 0
    for p in P_VALUES:
        rid = f"p{p}"
        engine.release_request(rid)
        med, best = median_ms(lambda p=p, rid=rid: engine.prefill(rid, token_ids=[7] * p))
        xs.append(p)
        ys.append(med)
        print(f"   P={p:>4}  median={med:8.3f} ms   best={best:8.3f} ms   "
              f"{med / p:7.4f} ms/token")
        engine.release_request(rid)
        counter += 1
    c0, c1, r2 = fit_line(xs, ys)
    print(f"   FIT  T(P) = {c0:.3f} + {c1:.5f} * P   (ms)   R2={r2:.4f}")
    print(f"   -> fixed cost per call c0 = {c0:.3f} ms ; marginal c1 = {c1:.5f} ms/token")
    share = c0 / (c0 + c1 * PROMPT_TOKENS) * 100.0
    print(f"   -> at P={PROMPT_TOKENS}: overhead share = {share:.1f}%  "
          f"(compute {100 - share:.1f}%)")

    # ---- decode: one token per call, steady state
    rid = "dec"
    engine.release_request(rid)
    engine.prefill(rid, token_ids=[7] * PROMPT_TOKENS)
    steps = []
    for i in range(DECODE_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.decode(rid, token_id=7)
        torch.cuda.synchronize()
        steps.append((time.perf_counter() - t0) * 1000.0)
    engine.release_request(rid)
    dec_med = statistics.median(steps[3:])
    dec_min = min(steps[3:])
    print(f"decode step (1 token): median={dec_med:.3f} ms  best={dec_min:.3f} ms  "
          f"(n={len(steps) - 3})")
    print(f"   bandwidth floor for this stage = {param_bytes / 1e6:.1f} MB / 200 GB/s "
          f"= {param_bytes / 200e9 * 1e3:.3f} ms")
    print(f"   measured / floor = {dec_med / (param_bytes / 200e9 * 1e3):.1f} x")
    print(f"   decode overhead share = "
          f"{max(0.0, (dec_med - param_bytes / 200e9 * 1e3) / dec_med) * 100:.1f}% "
          f"(if the stage were purely bandwidth bound)")

    await engine.shutdown()
    return {"layers": layers, "param_bytes": param_bytes, "c0_ms": c0, "c1_ms": c1,
            "r2": r2, "overhead_share_at_38": share, "decode_ms": dec_med}


async def measure_granularity():
    """K tokens through ONE call vs K calls of ONE token (the batching mechanism).

    Proxy for real batching: a batched decode of K requests reads the stage
    weights once and advances K sequences one token each, which is the same
    memory pattern as one call carrying K tokens (attention differs).
    """
    print("\n=== call granularity on ONE stage (layers 0..6): "
          "K tokens in 1 call vs K calls of 1 token ===")
    engine = await build("gran", (0, 6))
    print(f"   {'K':>4} {'1 call ms':>11} {'K calls ms':>12} {'speedup':>9} "
          f"{'ms/tok (1 call)':>17} {'ms/tok (K calls)':>18}")
    for k in (1, 2, 4, 8, 16, 32, 64):

        def one_call():
            engine.release_request("g1")
            engine.prefill("g1", token_ids=[7] * k)

        # warmup both forms
        for _ in range(2):
            one_call()
            for i in range(k):
                engine.release_request(f"gs{i}")
                engine.prefill(f"gs{i}", token_ids=[7])
        one_call()
        for i in range(k):
            engine.release_request(f"gs{i}")
            engine.prefill(f"gs{i}", token_ids=[7])

        one_ms = statistics.median(
            [timed(one_call) for _ in range(REPEATS)]
        )

        def k_calls():
            for i in range(k):
                rid = f"gk{i}"
                engine.prefill(rid, token_ids=[7])

        def k_calls_clean():
            k_calls()
            for i in range(k):
                engine.release_request(f"gk{i}")

        for _ in range(2):
            k_calls_clean()
        kc_ms = statistics.median([timed(k_calls_clean) for _ in range(REPEATS)])
        print(f"   {k:>4} {one_ms:>11.3f} {kc_ms:>12.3f} {kc_ms / one_ms:>8.2f}x "
              f"{one_ms / k:>17.4f} {kc_ms / k:>18.4f}")
        engine.release_request("g1")
    await engine.shutdown()


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


async def main():
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0)}")
    results = []
    for lr in [(0, 6), (0, 24)]:
        results.append(await measure_stage(lr))
    await measure_granularity()

    print("\n=== summary ===")
    for r in results:
        print(f"   {r['layers']:>2} layers: c0={r['c0_ms']:.3f} ms  "
              f"c1={r['c1_ms']:.5f} ms/token  R2={r['r2']:.4f}  "
              f"overhead@38tok={r['overhead_share_at_38']:.1f}%  "
              f"decode={r['decode_ms']:.3f} ms")

    small = [r for r in results if r["layers"] == 6][0]
    big = [r for r in results if r["layers"] == 24][0]
    # Per-layer values: differencing the two stage sizes cancels the embedding
    # table, which only stage 0 carries.
    per_layer_bytes = (big["param_bytes"] - small["param_bytes"]) / (24 - 6)
    c0_per_layer = big["c0_ms"] / 24
    c1_per_layer = ((big["c1_ms"] / 24) + (small["c1_ms"] / 6)) / 2
    print(f"\n   per-layer (0.5B, measured by differencing the two stages):")
    print(f"     weights  {per_layer_bytes / 1e6:.1f} MB/layer")
    print(f"     fixed    {c0_per_layer:.3f} ms per call per layer "
          f"(launch + python dispatch)")
    print(f"     marginal {c1_per_layer:.5f} ms per token per layer")
    print(f"   stage-0 extra (embedding table): "
          f"{(small['param_bytes'] - per_layer_bytes * 6) / 1e6:.1f} MB")

    # Qwen2.5-7B-Instruct 每层参数（从下载的 config.json 核对）：
    #   hidden=3584, intermediate=18944, num_attention_heads=28,
    #   num_key_value_heads=4, head_dim=128  -> GQA，注意 k/v 投影只有 4 个头
    #   attn = h*h + 2*h*(kv_heads*head_dim) + h*h
    #   mlp  = 3*h*intermediate
    per_layer_7b = (3584 * 3584 + 2 * 3584 * (4 * 128) + 3584 * 3584
                    + 3 * 3584 * 18944) * 2
    ratio = per_layer_7b / per_layer_bytes
    c1_7b = c1_per_layer * ratio
    print(f"\n   Qwen2.5-7B per-layer weights = {per_layer_7b / 1e6:.1f} MB "
          f"(28 layers, GQA 4 kv heads, bf16) -> {ratio:.1f}x of 0.5B")
    print(f"   break-even prompt tokens (framework overhead == model work):")
    print(f"     0.5B: {c0_per_layer / c1_per_layer:>6.0f} tokens")
    print(f"     7B:   {c0_per_layer / c1_7b:>6.0f} tokens")
    print(f"   projected overhead share in prefill:")
    for p in (38, 512, 4096):
        ov_small = c0_per_layer / (c0_per_layer + c1_per_layer * p)
        ov_7b = c0_per_layer / (c0_per_layer + c1_7b * p)
        print(f"     P={p:>5} tokens:  0.5B {ov_small * 100:5.1f}%   "
              f"7B {ov_7b * 100:5.1f}%")


asyncio.run(main())
