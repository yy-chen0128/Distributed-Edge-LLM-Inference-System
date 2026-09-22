"""读 7B checkpoint 的 safetensors 头，算每层真实字节 + 每段需要哪些分片文件。

为什么需要它：HF 的 safetensors 分片是按"约 3.6GB 大小"切的，和我们的流水线层边界
**不对齐**。四机部署时每台机器要么能访问整个 checkpoint，要么需要知道"我这几层其实
只落在哪几个文件里"。本脚本用 head 里的 data_offsets 直接算，不加载张量。

用法：
    python scripts/probe_7b_split_map.py --model .models/Qwen2.5-7B-Instruct \
        --split 12,4,4,8
"""

from __future__ import annotations

import argparse
import json
import os
import struct


def read_header(path: str) -> dict:
    """safetensors 头部：前 8 字节是 header 长度，之后是 JSON。"""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return {}
        n = struct.unpack("<Q", raw)[0]
        return json.loads(fh.read(n).decode("utf-8"))


def human(n: float) -> str:
    """十进制单位（kB/MB/GB）——与引擎上报的 param_bytes/1e6 同一口径。"""
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}TB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="12,4,4,8", help="各段层数，逗号分隔")
    args = parser.parse_args()

    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    total_layers = int(cfg["num_hidden_layers"])
    vocab = int(cfg["vocab_size"])
    hidden = int(cfg["hidden_size"])
    tie = bool(cfg.get("tie_word_embeddings", False))

    shards = sorted(f for f in os.listdir(args.model) if f.endswith(".safetensors"))
    print(f"model={args.model}")
    print(f"  layers={total_layers} hidden={hidden} vocab={vocab} tie={tie}")
    print(f"  shards={len(shards)}: " + ", ".join(shards))

    # tensor_name -> (shard, bytes)
    tensor_bytes: dict[str, tuple[str, int]] = {}
    shard_total: dict[str, int] = {}
    for shard in shards:
        header = read_header(os.path.join(args.model, shard))
        total = 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            size = int(end) - int(start)
            tensor_bytes[name] = (shard, size)
            total += size
        shard_total[shard] = total
    print("  shard sizes: " + ", ".join(f"{s}={human(b)}" for s, b in shard_total.items()))

    # per-layer bytes
    per_layer: dict[int, int] = {i: 0 for i in range(total_layers)}
    other: dict[str, int] = {}
    for name, (shard, size) in tensor_bytes.items():
        if name.startswith("model.layers."):
            idx = int(name.split(".")[2])
            per_layer[idx] += size
        else:
            other[name] = other.get(name, 0) + size

    nonzero = [b for b in per_layer.values() if b]
    print(f"\n  每层字节: {human(sum(nonzero) / max(1, len(nonzero)))}/层 "
          f"(min {human(min(nonzero))}, max {human(max(nonzero))})")
    print(f"  非层张量: " + ", ".join(f"{k}={human(v)}" for k, v in sorted(other.items())))

    # split
    split = [int(x) for x in args.split.split(",")]
    if sum(split) != total_layers:
        print(f"\n  !! split {split} 合计 {sum(split)} != {total_layers} 层，改用均分")
        split = [total_layers // len(split)] * len(split)
    print(f"\n  切分 {split}：")
    cursor = 0
    for i, count in enumerate(split):
        start, end = cursor, cursor + count
        weight = sum(per_layer[j] for j in range(start, end))
        fixed = 0
        fixed_names = []
        if i == 0:
            for key in ("model.embed_tokens.weight",):
                if key in tensor_bytes:
                    fixed += tensor_bytes[key][1]
                    fixed_names.append(key)
        if i == len(split) - 1:
            names = ["model.norm.weight", "lm_head.weight"]
            for key in names:
                if key in tensor_bytes:
                    fixed += tensor_bytes[key][1]
                    fixed_names.append(key)
        needed = set()
        for j in range(start, end):
            for name, (shard, _) in tensor_bytes.items():
                if name.startswith(f"model.layers.{j}."):
                    needed.add(shard)
        for key in fixed_names:
            needed.add(tensor_bytes[key][0])
        cursor = end
        print(f"    段{i} 层[{start},{end}) {count:>2}层 权重={human(weight):>8} "
              f"固定={human(fixed):>8} 合计={human(weight + fixed):>8} "
              f"| 需要分片({len(needed)}): {', '.join(sorted(needed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
