"""Fleet planner: turn N per-machine readiness reports into a deployment decision.

    python scripts/fleet_plan_from_readiness.py node_readiness_*.json

The heterogeneity rules it applies (all three are consequences of how the stack
works, not preferences):

  1. STACK VERSION is capped by the machine with the LOWEST driver CUDA version,
     because one wheel set is installed on every machine.
  2. MODEL TIER is capped by the machine with the LEAST usable VRAM.
  3. The LAYER SPLIT is EVEN in layer count, because vLLM distributes layers evenly
     across PP ranks -- it does not weight by VRAM. So a fleet whose smallest machine
     cannot hold its even share simply cannot run that model, and the options are:
     quantise, drop the smallest machine, or use a smaller model.

It also lists the hard prerequisites that must hold on every participating machine
(mirrored WSL networking, reachable peer, working CUDA in the venv, transformers<5,
free ports) and separates those blockers from mere warnings.

ASCII output.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Optional

SPLIT_MAP = {
    "cuda": 0.0,
    "vllm>=0.11 (torch 2.13+/cu130)": 13.0,
    "vllm 0.10.2 (torch 2.8/cu128)": 12.8,
    "vllm 0.10.1 / 0.9.2 (torch 2.7.x/cu126)": 12.6,
    "vllm 0.8.5 (torch 2.6/cu124)": 12.4,
}

# tier -> (per-layer GB for 7B, per-layer GB for a 3B-class model is not modelled here)
TIERS = {
    "7b_fp16": {"per_layer_gb": 0.4661, "vocab_gb": 1.09, "label": "7B fp16"},
    "7b_int4": {"per_layer_gb": 0.1280, "vocab_gb": 1.09, "label": "7B AWQ/GPTQ int4"},
    "3b_fp16": {"per_layer_gb": 0.1900, "vocab_gb": 0.55, "label": "3B fp16 (approx)"},
}
RESERVE_GB = 0.8          # activations, KV, fragmentation
LAYERS_7B = 28


def cuda_of(facts: dict) -> float:
    return float(facts.get("cuda_version") or 0.0)


def usable_of(facts: dict) -> float:
    return float(facts.get("vram_usable_gb") or 0.0)


def even_split(layers: int, parts: int) -> list[int]:
    """vLLM-style even layer distribution (remainder to the earlier ranks)."""
    base, rem = divmod(layers, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def stage_cost_gb(tier: str, layers: int, is_first: bool, is_last: bool) -> float:
    spec = TIERS[tier]
    cost = layers * spec["per_layer_gb"]
    if is_first or is_last:
        cost += spec["vocab_gb"]        # embedding (first) / output head (last)
    return cost


def evaluate(nodes: list[dict], tier: str, pp: int) -> Optional[dict]:
    """Can `pp` of the nodes run 7B in `tier` with vLLM's even layer split?"""
    if pp < 1 or pp > len(nodes):
        return None
    spec = TIERS[tier]
    # pick the pp machines with the most usable VRAM (drop the weakest first)
    chosen = sorted(nodes, key=lambda n: -usable_of(n["facts"]))[:pp]
    chosen.sort(key=lambda n: n["facts"]["hostname"])       # deterministic order
    per = even_split(LAYERS_7B, pp)
    stages = []
    ok = True
    for i, node in enumerate(chosen):
        cost = stage_cost_gb(tier, per[i], i == 0, i == pp - 1)
        budget = usable_of(node["facts"])
        fits = cost <= budget
        ok = ok and fits
        stages.append({
            "hostname": node["facts"]["hostname"],
            "layers": per[i],
            "cost_gb": round(cost, 2),
            "budget_gb": round(budget, 2),
            "fits": fits,
            "vram_gb": node["facts"].get("vram_gb"),
            "cuda": node["facts"].get("cuda_version"),
        })
    dropped = [n["facts"]["hostname"] for n in nodes if n not in chosen]
    return {"tier": tier, "tier_label": spec["label"], "pp": pp, "feasible": ok,
            "stages": stages, "dropped": dropped,
            "max_stage_gb": round(max(s["cost_gb"] for s in stages), 2)}


def blockers_of(node: dict) -> list[str]:
    """Hard prerequisites. Anything here means the machine cannot join as-is."""
    facts, verdicts = node["facts"], node.get("verdicts", {})
    out = []
    if verdicts.get("wsl-networking") == "FAIL":
        out.append("WSL not in mirrored networking mode -> unreachable from peers")
    if verdicts.get("stack-ceiling") == "FAIL":
        out.append(f"driver CUDA {facts.get('cuda_version')} < 12.4 -> no usable wheel set")
    if verdicts.get("gpu") == "FAIL":
        out.append("no visible NVIDIA GPU")
    if verdicts.get("transformers-pin") == "FAIL":
        out.append(f"transformers {facts.get('transformers')} >= 5 breaks vLLM 0.10.1")
    venv = facts.get("vllm") or {}
    if isinstance(venv, dict) and venv.get("cuda_available") is False:
        out.append("torch in ~/venvs/vllm reports cuda_available=False")
    if verdicts.get("peer-ping") == "FAIL":
        out.append(f"cannot ping the head node ({facts.get('peer_ping_loss_pct')}% loss)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="node_readiness_*.json files")
    ap.add_argument("--layers", type=int, default=LAYERS_7B)
    args = ap.parse_args()

    nodes = []
    for path in args.reports:
        with open(path, encoding="utf-8") as fh:
            nodes.append(json.load(fh))

    print("=" * 96)
    print(f"FLEET PLAN from {len(nodes)} readiness report(s)")
    print("=" * 96)

    # ---- per-machine table (the heterogeneity facts)
    hdr = (f"{'host':<18}{'gpu':<26}{'vram':>6}{'usable':>8}{'drv':>8}{'cuda':>7}"
           f"{'sm':>6}  {'max stack':<42}")
    print(hdr)
    print("-" * len(hdr))
    for node in nodes:
        f = node["facts"]
        print(f"{f.get('hostname','?'):<18}{str(f.get('gpu','?'))[:24]:<26}"
              f"{f.get('vram_gb',0):>6}{f.get('vram_usable_gb',0):>8}"
              f"{str(f.get('driver_version','?')):>8}{f.get('cuda_version',0):>7}"
              f"{str(f.get('compute_cap','?')):>6}  {str(f.get('max_stack')):<42}")

    # ---- rule 1: the stack ceiling is the minimum
    cudas = [cuda_of(n["facts"]) for n in nodes]
    min_cuda = min(cudas) if cudas else 0.0
    chosen_stack = None
    for label, need in sorted(SPLIT_MAP.items(), key=lambda kv: -kv[1]):
        if label == "cuda":
            continue
        if min_cuda >= need:
            chosen_stack = (need, label)
            break
    print()
    print("RULE 1 - stack version (capped by the oldest driver):")
    print(f"  lowest driver CUDA across the fleet = {min_cuda}")
    for node in nodes:
        flag = "" if cuda_of(node['facts']) == min_cuda else "   (newer than the floor)"
        print(f"    {node['facts']['hostname']:<18} CUDA {node['facts'].get('cuda_version')}{flag}")
    if chosen_stack:
        print(f"  => install on ALL machines: {chosen_stack[1]}")
    else:
        print("  => NO usable stack: every machine needs a driver upgrade (>= CUDA 12.4)")

    # ---- rule 2+3: model tier x PP size, with the EVEN split vLLM will actually use
    print()
    print("RULE 2+3 - model tier x PP size (vLLM splits layers EVENLY, it ignores VRAM):")
    options = []
    for tier in ("7b_fp16", "7b_int4", "3b_fp16"):
        for pp in range(len(nodes), 1, -1):
            result = evaluate(nodes, tier, pp)
            if result:
                options.append(result)
    feasible = [o for o in options if o["feasible"]]
    seen_tiers = set()
    if feasible:
        for opt in feasible:
            key = (opt["tier"], opt["pp"])
            if key in seen_tiers:
                continue
            seen_tiers.add(key)
            note = f"  (drops: {', '.join(opt['dropped'])})" if opt["dropped"] else ""
            print(f"  FEASIBLE  {opt['tier_label']:<22} PP={opt['pp']}  "
                  f"max stage {opt['max_stage_gb']:.2f} GB{note}")
            if opt["pp"] == len(nodes) and opt["tier"] == "7b_fp16":
                pass
    else:
        print("  NO feasible option for 7B with the current fleet.")
        print("  Options, in order of least damage:")
        print("    - use int4 (AWQ/GPTQ) for 7B, or a smaller model (0.5B/1.5B/3B)")
        print("    - reduce PP (drop the smallest machine from the pipeline)")
        print("    - free VRAM on the weakest machine (close other GPU apps)")

    # the recommended option: most machines first, then the best tier
    if feasible:
        best = sorted(feasible, key=lambda o: (-o["pp"], 0 if o["tier"] == "7b_fp16" else 1))[0]
        print()
        print("RECOMMENDATION:")
        print(f"  {best['tier_label']}, PP={best['pp']}, even layer split:")
        for st in best["stages"]:
            print(f"    {st['hostname']:<18} {st['layers']:>2} layers  "
                  f"{st['cost_gb']:>5.2f} GB / {st['budget_gb']:>5.2f} GB available  "
                  f"{'ok' if st['fits'] else 'OVER'}")
        if best["dropped"]:
            print(f"  machines left out: {', '.join(best['dropped'])}")

    # ---- hard prerequisites
    print()
    print("HARD PREREQUISITES (a machine with blockers cannot join as-is):")
    any_blocker = False
    for node in nodes:
        bl = blockers_of(node)
        host = node["facts"]["hostname"]
        if bl:
            any_blocker = True
            print(f"  {host}:")
            for b in bl:
                print(f"    - {b}")
        else:
            print(f"  {host}: none")
    if not any_blocker:
        print("  (all machines pass)")

    # ---- heterogeneity summary
    print()
    print("HETEROGENEITY SUMMARY:")
    vrams = {n['facts']['hostname']: n['facts'].get('vram_gb') for n in nodes}
    caps = {n['facts']['hostname']: n['facts'].get('compute_cap') for n in nodes}
    drvs = {n['facts']['hostname']: n['facts'].get('driver_version') for n in nodes}
    print(f"  VRAM      : {vrams}   -> {'heterogeneous (exploitable only by an engine that weights splits; vLLM does not)' if len(set(vrams.values())) > 1 else 'uniform'}")
    print(f"  GPU arch  : {caps}   -> {'heterogeneous (decides FP8/kernels, not blocking)' if len(set(caps.values())) > 1 else 'uniform'}")
    print(f"  driver    : {drvs}   -> {'heterogeneous (NOT exploitable; the floor caps the whole fleet)' if len(set(drvs.values())) > 1 else 'uniform'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
