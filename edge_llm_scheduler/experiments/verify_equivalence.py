"""分段执行 vs 整体执行的数值一致性验证（阶段 D 的验收证据）。

做法：同一份真实权重、同一 dtype、同样的贪心解码路径下，
1) 整体执行：一次前向跑完 24 层；
2) 分段执行：把 24 层按给定区间切给 N 个 `HFLayeredEngine`，hidden states 在
   内存里按 stage 顺序传递（本地验证数值路径；跨机由 run_real_pipeline.py 走 TCP）。
比较：每步采样出的 token id 是否完全一致，以及末段 logits 的最大绝对差。

用法：
    python -m edge_llm_scheduler.experiments.verify_equivalence \
        --model .models/Qwen2.5-0.5B-Instruct --prompt "介绍一下你自己" --max-tokens 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from ..backends.hf_layered_engine import HFLayeredEngine
from ..core.types import StageAssignment


def split_layers(total: int, stages: int, weights: list[float] | None = None) -> list[tuple[int, int]]:
    """按权重（默认平均）切出连续且完整覆盖的层区间。"""
    weights = weights or [1.0] * stages
    total_w = sum(weights)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for idx, weight in enumerate(weights):
        remaining_stages = stages - idx - 1
        remaining_layers = total - cursor
        if remaining_stages == 0:
            end = total
        else:
            share = max(1, round(total * weight / total_w))
            share = min(share, remaining_layers - remaining_stages)
            end = cursor + share
        ranges.append((cursor, end))
        cursor = end
    assert ranges[0][0] == 0 and ranges[-1][1] == total, ranges
    return ranges


class _Monolithic:
    """整体执行的参照实现：同一个 HF 模型，一次跑完所有层。"""

    def __init__(self, model_path: str, dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        torch_dtype = getattr(torch, dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype)
        except TypeError:  # 老版本 transformers
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.model.eval()
        self.cache = None

    def step(self, input_ids, past=None):
        # logits_to_keep=1：只算最后一个位置的 logits，才与分段实现公平可比
        try:
            out = self.model(input_ids=input_ids, past_key_values=past, use_cache=True,
                             logits_to_keep=1)
        except TypeError:  # 老版本没有该参数
            out = self.model(input_ids=input_ids, past_key_values=past, use_cache=True)
        return out.logits[:, -1, :], out.past_key_values


class _Sharded:
    """分段执行的参照实现：N 个 HFLayeredEngine 串起来（进程内）。"""

    def __init__(self, model_path: str, ranges: list[tuple[int, int]], dtype: str = "float32"):
        self.engines = [
            HFLayeredEngine(f"stage{i}", model_path, device="cpu", dtype=dtype, logger=lambda *_: None)
            for i in range(len(ranges))
        ]
        self.ranges = ranges

    async def prepare(self, epoch: int = 0) -> None:
        for engine, (start, end) in zip(self.engines, self.ranges):
            await engine.prepare_epoch(StageAssignment(epoch, engine.node_id, (start, end)))
            await engine.activate_epoch(epoch)

    def prefill(self, request_id: str, tokens: list[int]) -> tuple[int, float]:
        hidden = None
        token = None
        logits_last = None
        started = time.perf_counter()
        for idx, engine in enumerate(self.engines):
            out = engine.prefill(request_id, token_ids=tokens if idx == 0 else None, hidden=hidden)
            hidden = out.hidden
            if out.token_id is not None:
                token = out.token_id
                logits_last = out.logits_last
        return token, logits_last, (time.perf_counter() - started) * 1000.0

    def decode(self, request_id: str, token: int) -> tuple[int, float]:
        hidden = None
        next_token = None
        logits_last = None
        started = time.perf_counter()
        for idx, engine in enumerate(self.engines):
            out = engine.decode(request_id, token_id=token if idx == 0 else None, hidden=hidden)
            hidden = out.hidden
            if out.token_id is not None:
                next_token = out.token_id
                logits_last = out.logits_last
        return next_token, logits_last, (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="用一句话解释什么是张量并行。")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text, return_tensors=None)["input_ids"]
    print(f"prompt tokens: {len(tokens)} 层数: 24 分段: {args.stages} dtype={args.dtype}")

    ranges = split_layers(24, args.stages)
    print("层区间:", ranges)

    # 1) 整体执行
    mono = _Monolithic(args.model, args.dtype)
    started = time.perf_counter()
    logits, cache = mono.step(mono.torch.tensor([tokens]))
    mono_tokens = [int(mono.torch.argmax(logits, dim=-1).item())]
    mono_logits = [logits.clone()]
    for _ in range(args.max_tokens - 1):
        logits, cache = mono.step(mono.torch.tensor([[mono_tokens[-1]]]), past=cache)
        mono_tokens.append(int(mono.torch.argmax(logits, dim=-1).item()))
        mono_logits.append(logits.clone())
    mono_ms = (time.perf_counter() - started) * 1000.0

    # 2) 分段执行
    sharded = _Sharded(args.model, ranges, args.dtype)
    asyncio.run(sharded.prepare(0))
    first_token, first_logits, shard_ms = sharded.prefill("req-1", tokens)
    seq = [first_token]
    shard_logits_list = [first_logits]
    step_ms = [shard_ms]
    for _ in range(args.max_tokens - 1):
        nxt, lg, ms = sharded.decode("req-1", seq[-1])
        seq.append(nxt)
        shard_logits_list.append(lg)
        step_ms.append(ms)

    diffs = []
    for mono_step, shard_step in zip(mono_logits, shard_logits_list):
        if shard_step is None:
            continue
        diffs.append(float((mono_step - shard_step).abs().max().item()))
    max_diff = max(diffs) if diffs else float("nan")

    text_mono = tokenizer.decode(mono_tokens, skip_special_tokens=True)
    text_shard = tokenizer.decode(seq, skip_special_tokens=True)
    metrics = {
        "prompt_tokens": len(tokens),
        "stages": args.stages,
        "ranges": ranges,
        "dtype": args.dtype,
        "mono_tokens": mono_tokens,
        "sharded_tokens": seq,
        "tokens_equal": mono_tokens == seq,
        "max_logit_diff": max_diff,
        "per_step_logit_diff": diffs,
        "mono_total_ms": mono_ms,
        "sharded_prefill_ms": shard_ms,
        "sharded_decode_ms": step_ms[1:],
        "mono_text": text_mono,
        "sharded_text": text_shard,
        "param_bytes_per_stage": [
            (e._active_stage().param_bytes if e._active_stage() else 0) for e in sharded.engines
        ],
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("token 完全一致:" , metrics["tokens_equal"], "  logits 最大绝对差:", max_diff)
    print("整体执行文本:", text_mono)
    print("分段执行文本:", text_shard)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
    return 0 if metrics["tokens_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
