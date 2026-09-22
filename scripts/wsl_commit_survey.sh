#!/usr/bin/env bash
# Commit the engine-selection update + the survey doc. ASCII ONLY.
set -eu
cd "$HOME/pair"

echo "=== sanity: clones untracked ==="
git status --short | grep -E '(ollama|lmstudio|lms)/' && { echo ABORT; exit 1; } || echo ok

git add -A
echo
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: fold the multi-machine engine survey into the selection doc

Adds the full survey (docs/research/multi-machine-inference-engine-survey-2026-09.md,
evidence-graded V/I/U with URLs) and folds its findings into inference-engine-selection.md.

Corrections to the previous version:
- llama.cpp: -sm/--split-mode DEFAULT is `layer` ("split layers and KV across GPUs
  (pipelined)"), not tensor; KV stays on the owning device. But it is NOT a true
  pipeline: no cross-machine stage overlap, one token traverses all devices, so our
  "many requests in flight" idea cannot be expressed. -sm tensor is EXPERIMENTAL.
- vLLM: my claim that PP requires equal layer counts per stage was WRONG. The official
  docs have an "uneven GPU splits" entry, and SGLang has SGLANG_PP_LAYER_PARTITION.
  The real blockers are: official docs require an identical execution environment across
  nodes ("to hide host heterogeneity"), no runtime re-split, no elasticity (fault
  tolerance delegated to Ray). Conclusion (not usable as our engine) is unchanged.

New evidence added:
- llama.cpp node loss = GGML_ABORT in current master; the unmerged PR #26724 only turns
  the crash into "device permanently unusable" + 503, quoting "a lost connection can
  never be recovered". Verified master still has RPC_STATUS_ASSERT as of 2026-09-22.
- No official llama.cpp RPC performance numbers; a third-party 10GbE benchmark shows
  7B Q4_K_M decode 76.1 -> 52.7 tok/s and concludes "RPC is for capacity, not speed".
- Mesh-LLM (Apache-2.0): cross-machine contiguous layer stages, latency-aware planner,
  arbitrary half-open ranges, but topology revocation only -- no in-flight resume.
  Worth a 1-2 day spike.
- Excluded with source-level reasons: distributed-llama (equal tensor parallelism,
  assert(d % nNodes == 0); measured 4-node 1GbE: 11.00 -> 1.85 t/s), ktransformers /
  PowerInfer (single machine; PowerInfer-2 has no public repo), LocalAI, Ghostlink,
  MLC-LLM (docs 404), TPI-LLM (allreduce bottleneck is latency, not bandwidth).
- Paper-level references for R1/R2: prima.cpp (674 vs 20,848 ms/token = 15-31x for
  heterogeneous splits, but code URL 404), SpotServe (stateful recovery, code not
  released), DynaPipe (async KV migration), LUMEN (needs 80-160 GB host RAM per
  worker), PipeBoost (single machine).
- Headline: R2 (in-flight recovery on consumer multi-machine hardware) is a BLANK --
  every system that does it (SpotServe/LUMEN/PipeBoost/DynaPipe) is datacenter or
  single-machine. That asymmetry is the publishable gap.

New section 6.5 collects the sentences we can quote directly in the paper.
Also flags "ServerlessLLM Flex" as unverifiable -- do not cite it.
EOF
git commit -F "$MSGFILE"
rm -f "$MSGFILE"

echo
git log --oneline -3
echo
echo "=== push ==="
git push origin main
git status -sb | head -2
git ls-files | wc -l
