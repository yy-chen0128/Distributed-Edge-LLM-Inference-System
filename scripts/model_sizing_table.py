"""拉取候选模型的真实 config，算出显存/KV 需求表（部署选型用）。"""

from __future__ import annotations

import json
import urllib.request

REPOS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
]
ENDPOINT = "https://hf-mirror.com"


def fetch(repo: str) -> dict:
    url = f"{ENDPOINT}/{repo}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    header = (f"{'model':38} {'lay':>4} {'hid':>5} {'kv':>3} {'hd':>4} "
              f"{'vocab':>7} {'exp':>5} {'params(B)':>9} {'fp16 GB':>8} {'int4 GB':>8} "
              f"{'KV KB/tok':>9} {'KV GB@8k':>8}")
    print(header)
    print("-" * len(header))
    for repo in REPOS:
        try:
            cfg = fetch(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"{repo:38} 拉取失败: {str(exc)[:40]}")
            continue
        layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layers"))
        hidden = int(cfg.get("hidden_size") or cfg.get("d_model"))
        heads = int(cfg.get("num_attention_heads") or 0)
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        head_dim = int(cfg.get("head_dim") or (hidden // heads if heads else 0))
        vocab = int(cfg.get("vocab_size") or 0)
        experts = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or 0)
        inter = int(cfg.get("intermediate_size") or cfg.get("moe_intermediate_size") or 0)
        attn = hidden * hidden + 2 * hidden * (kv_heads * head_dim) + hidden * hidden
        if experts:
            mlp = inter * hidden * 3 * experts
            # MoE 总参数里专家占绝大部分
            params = layers * (attn + mlp) + 2 * vocab * hidden
        else:
            mlp = inter * hidden * 3
            params = layers * (attn + mlp) + 2 * vocab * hidden
        kv_per_token = 2 * layers * kv_heads * head_dim * 2  # fp16
        print(f"{repo:38} {layers:4d} {hidden:5d} {kv_heads:3d} {head_dim:4d} "
              f"{vocab:7d} {experts:5d} {params/1e9:9.2f} {params*2/1e9:8.2f} "
              f"{params*0.55/1e9:8.2f} {kv_per_token/1024:9.1f} "
              f"{kv_per_token*8192/1e9:8.2f}")


if __name__ == "__main__":
    main()
