#!/usr/bin/env bash
# Confirm the new install is pulling torch 2.7.x (cu126), not cu13x. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
LOG=.models/logs/vllm10_install.log

echo "=== torch / nvidia lines in the NEW log ==="
grep -E 'Downloading (torch|nvidia_)' "$LOG" 2>/dev/null | tail -12
echo
echo "=== torch version being collected ==="
grep -oE 'torch-[0-9]+\.[0-9]+\.[0-9]+[^ ]*\.whl' "$LOG" 2>/dev/null | sort -u | head -5
echo
echo "=== vllm wheel ==="
grep -oE 'vllm-[0-9.]+[^ ]*\.whl' "$LOG" 2>/dev/null | sort -u | head -3
echo
echo "=== progress ==="
echo "venv: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)"
grep -c 'Downloading' "$LOG" 2>/dev/null | sed 's/^/  downloads so far: /'
echo "--- last 6 lines ---"
tail -6 "$LOG"
echo
echo "=== commmit the doc change ==="
git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -m "docs: three kinds of heterogeneity and the driver/CUDA decision rule

Answers whether differing drivers across the four machines are a problem.
They need not be identical: NCCL/CUDA collectives do not require the same driver.
What they must all satisfy is the same LOWER BOUND, because we install one wheel
set for the whole cluster, and the hard rule is that the driver's reported CUDA
version must be >= the wheel's CUDA version.

The interconnect doc now separates three kinds of heterogeneity whose effects are
completely different in nature:
1. VRAM/compute (8/6/4/4 GB) -- partially exploitable via capacity-weighted
   splitting (our engine does this; vLLM just splits evenly);
2. GPU architecture (sm86 vs sm89) -- not exploitable but not blocking; it decides
   which kernels/quantisation formats exist (FP8 needs sm89+);
3. driver/CUDA version -- NOT exploitable at all, only a shared floor.
For (3) the doc gives the mapping (cu124 -> driver >= 12.4, cu126 -> >= 12.6,
cu128 -> >= 12.8, cu130 -> >= 13.0), states that the oldest machine sets the
cluster-wide ceiling (so with one 566.24 machine the fleet is capped at vLLM
0.10.1), notes that upgrading the driver is a Windows-side action, and records the
one command each machine must run to report its CUDA version before we start.

Also notes the flip side: the self-built engine only needs torch 2.6+cu124
(CUDA >= 12.4), so it tolerates older drivers better than current vLLM." 2>&1 | tail -2
git -c safe.directory='*' push origin main 2>&1 | tail -2
