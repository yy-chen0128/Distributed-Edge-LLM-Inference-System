"""判断候选模型能否被 N 台笔记本按层切开装下（PP 选型用）。

与 `model_sizing_table.py` 的区别：这张表专门服务"跨机 PP"决策，因此额外算三件事：
1. **每层参数字节**（决定某台机器能分到几层）；
2. **两端固定开销**：首段要装 embedding、末段要装 lm_head；若 `tie_word_embeddings=True`
   且 checkpoint 里没有独立 lm_head，则**末段还要再装一份词表**（Qwen2.5 就是这种）；
3. **每 token 的 KV 字节**（决定长上下文能开到多少）。

用法（各机都能跑，走 HF 镜像）：
    python scripts/model_pp_fit.py --stages 4 --vram 8,6,4,4
    python scripts/model_pp_fit.py --models Qwen/Qwen2.5-7B-Instruct --stages 4 --vram 8,8,8,8
"""

from __future__ import annotations

import argparse
import json
import urllib.request

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
]
ENDPOINT = "https://hf-mirror.com"


def fetch_config(repo: str, endpoint: str, timeout: float) -> dict:
    url = f"{endpoint}/{repo}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def analyse(repo: str, cfg: dict) -> dict:
    layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layers") or 0)
    hidden = int(cfg.get("hidden_size") or cfg.get("d_model") or 0)
    heads = int(cfg.get("num_attention_heads") or 0)
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or (hidden // heads if heads else 0))
    vocab = int(cfg.get("vocab_size") or 0)
    experts = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or 0)
    inter = int(
        cfg.get("moe_intermediate_size") if experts else cfg.get("intermediate_size") or 0
    ) or int(cfg.get("intermediate_size") or 0)
    tie = bool(cfg.get("tie_word_embeddings", False))

    # 每层参数：注意力 + MLP（MoE 只算"单专家"体量，另列总专家数）
    attn = hidden * hidden + 2 * hidden * (kv_heads * head_dim) + hidden * hidden
    mlp = inter * hidden * 3
    per_layer = attn + mlp
    embed = vocab * hidden

    return {
        "repo": repo,
        "layers": layers,
        "hidden": hidden,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab": vocab,
        "experts": experts,
        "expert_inter": inter if experts else 0,
        "tie_embeddings": tie,
        "per_layer_params": per_layer,
        "per_layer_fp16_gb": per_layer * 2 / 1e9,
        "per_layer_int4_gb": per_layer * 0.55 / 1e9,
        "embed_fp16_gb": embed * 2 / 1e9,
        "lm_head_extra_fp16_gb": 0.0 if (not tie) else embed * 2 / 1e9,
        # 每 token 的 KV（fp16）
        "kv_kb_per_token_fp16": 2 * layers * kv_heads * head_dim * 2 / 1024,
    }


def plan_stages(info: dict, stages: int, vram: list[float], dtype: str) -> dict:
    """平均切层，算出每段的权重占用与余量（不追求最优切分，只看"装不装得下"）。"""
    per_layer = info["per_layer_fp16_gb"] if dtype == "fp16" else info["per_layer_int4_gb"]
    base = info["layers"] // stages
    counts = [base] * stages
    for i in range(info["layers"] % stages):
        counts[i] += 1
    rows = []
    for i, cnt in enumerate(counts):
        weight = cnt * per_layer
        fixed = 0.0
        if i == 0:
            fixed += info["embed_fp16_gb"]
        if i == stages - 1:
            fixed += info["lm_head_extra_fp16_gb"]
            if not info["tie_embeddings"]:
                fixed += info["embed_fp16_gb"]  # 未绑定 → 独立 lm_head，与词表同量级
        need = weight + fixed
        rows.append(
            {
                "stage": i,
                "layers": cnt,
                "weight_gb": round(weight, 2),
                "fixed_gb": round(fixed, 2),
                "total_gb": round(need, 2),
                "budget_gb": vram[i],
                "fits": need + 0.8 <= vram[i],  # 留 0.8GB 给 activation/context/fragmentation
            }
        )
    return {"rows": rows, "all_fit": all(r["fits"] for r in rows)}


def main() -> None:
    # Windows 控制台默认 GBK，中文/符号会炸；统一改成 UTF-8
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--vram", default="8,6,4,4", help="各机可用显存 GB，按机器列出")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "int4"])
    parser.add_argument("--kv-ctx", type=int, default=4096)
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()

    vram = [float(x) for x in args.vram.split(",")]
    if len(vram) != args.stages:
        raise SystemExit(f"--vram 数量({len(vram)}) 与 --stages({args.stages}) 不一致")

    for repo in args.models:
        try:
            cfg = fetch_config(repo, args.endpoint, 30)
        except Exception as exc:  # noqa: BLE001
            print(f"\n### {repo}\n  拉取 config 失败: {str(exc)[:60]}")
            continue
        info = analyse(repo, cfg)
        print(f"\n### {repo}")
        print(
            f"  层数={info['layers']} hidden={info['hidden']} kv_heads={info['kv_heads']} "
            f"head_dim={info['head_dim']} vocab={info['vocab']} "
            f"专家={info['experts'] or '稠密'} tie_embeddings={info['tie_embeddings']}"
        )
        print(
            f"  每层权重: fp16 {info['per_layer_fp16_gb']:.3f}GB / int4 {info['per_layer_int4_gb']:.3f}GB"
            f" | 词表(fp16) {info['embed_fp16_gb']:.2f}GB"
            f" | KV {info['kv_kb_per_token_fp16']:.1f}KB/token"
            f" ({info['kv_kb_per_token_fp16'] * args.kv_ctx / 1e6:.2f}GB @{args.kv_ctx}ctx)"
        )
        plan = plan_stages(info, args.stages, vram, args.dtype)
        verb = "[装得下]" if plan["all_fit"] else "[装不下]"
        print(f"  {args.stages} 段平均切层（dtype={args.dtype}，预算 {vram}）：{verb}")
        for row in plan["rows"]:
            flag = "ok " if row["fits"] else "OVER"
            print(
                f"    段{row['stage']} 层数={row['layers']:>2} 权重={row['weight_gb']:.2f}GB "
                f"固定={row['fixed_gb']:.2f}GB 合计={row['total_gb']:.2f}GB "
                f"/ 预算{row['budget_gb']:.1f}GB  [{flag}]"
            )
        if info["experts"]:
            print(
                f"    MoE 注：单专家 intermediate={info['expert_inter']}，共 {info['experts']} 专家；"
                f"专家并行时每台只放 1/{args.stages} 的专家"
            )


if __name__ == "__main__":
    main()
