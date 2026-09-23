#!/usr/bin/env bash
# Re-verify the readiness check after the transformers-pin fix, then commit.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
PY="$HOME/venvs/pair/bin/python"
OUT=.models/logs/readiness

echo "=== compile + rerun the self-check ==="
"$PY" -m py_compile scripts/check_node_readiness.py || exit 1
"$PY" scripts/check_node_readiness.py --json-out "$OUT/node_readiness_$(hostname).json" 2>&1 \
  | grep -E '^\[|^FAIL|^### |transformers-pin|lmcache' | head -25

echo
echo "=== commit ==="
git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -F - <<'MSG'
feat: per-node readiness self-check + heterogeneity-aware fleet planner

Two scripts, because "can this machine join" and "what is the fleet's best option"
are different questions with different owners.

scripts/check_node_readiness.py (run inside WSL on every candidate machine):
- reports the raw facts and a verdict per item: GPU model, VRAM, driver version, the
  driver's CUDA version, compute capability, WSL/OS, python, disk, pip config, free
  ports, and the versions inside each venv (torch/vllm/ray/lmcache/transformers)
- applies the stack rule: the driver's CUDA version decides the newest usable wheel
  set (cu124 >= 12.4, cu126 >= 12.6, cu128 >= 12.8, cu130 >= 13.0)
- flags the WSL mirrored-networking requirement (without it no peer can reach this
  machine, so Ray/vLLM PP cannot work)
- flags the pip http-mirror trap that stalled our own install
- checks the transformers<5 pin, but ONLY in the venv that has vLLM (our own
  engine's venv legitimately uses transformers 5.x -- flagging it there was a false
  positive in the first version)
- emits the heterogeneity numbers the controller needs: usable VRAM, how many 7B
  layers fit in fp16 vs int4, the recommended model tier, and per local model the
  per-layer GB, vocabulary GB and KV KB/token
- optionally tests a peer: ping loss and TCP reachability of 6379/8000/8100/22/2222
- writes node_readiness_<host>.json and prints a "send these back" block

scripts/fleet_plan_from_readiness.py (run by the controller over N reports):
- RULE 1 stack version is capped by the machine with the lowest driver CUDA version,
  because one wheel set is installed everywhere
- RULE 2+3 model tier and PP size: vLLM distributes layers EVENLY across PP ranks and
  ignores VRAM, so for each tier x PP size it checks whether every stage fits
  (including the first/last stage's vocabulary/output-head cost) and reports the
  feasible options plus a recommendation, or says there is none and lists what to do
  instead (quantise, drop the smallest machine, or a smaller model)
- lists HARD PREREQUISITES per machine (mirrored networking, reachable peer, working
  CUDA in the venv, transformers<5, ports) separately from warnings
- summarises the three kinds of heterogeneity with the verdict that matters: VRAM
  heterogeneity is partially exploitable (not by vLLM), GPU-arch heterogeneity is
  informative but not blocking, driver heterogeneity is not exploitable at all and
  caps the whole fleet

Verified by running the self-check on this machine (8 GB, driver 566.24/CUDA 12.7,
sm89, mirrored networking, vLLM 0.10.1 + torch 2.7.1+cu126 working) and the planner
on four synthetic laptops (8/6/4/4 GB with CUDA 12.7/12.6/12.4/12.4). The planner
correctly concluded: the fleet is capped at the cu124 stack by the two 12.4 machines,
7B fp16 is infeasible at any PP >= 2 on this fleet, 7B AWQ int4 at PP=4 is feasible,
and one synthetic machine was rejected for lacking mirrored networking.
MSG
git -c safe.directory='*' push origin main 2>&1 | tail -2
echo "pushed $(git rev-parse --short HEAD)"
