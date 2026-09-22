#!/usr/bin/env bash
# Commit: request-position reporting, 7B shard/layer map, recovery requirements.
# ASCII ONLY.
set -eu
cd "$HOME/pair"

git add -A
git diff --cached --stat

MSG=$(cat <<'EOF'
feat: report per-request progress; map 7B layers to HF shards

Engine:
- HFLayeredEngine.request_positions() reports, per open request, how many tokens
  have been processed (the cache sequence length). This is the reading that
  interrupt recovery needs, and the engine can answer it directly -- no need to
  read KV tensors. Verified on GPU: 38 after a 38-token prefill, 43 after four
  more decode steps, and a second request reports its own count independently.
  status() now includes it.
- Caveat recorded: only a live process can answer, so the control plane must
  count tokens itself as the authoritative source; the engine reading is for
  verification and for graceful shutdown reports.
- tests/test_kv_accounting.py: two more tests (per-request progress, epoch
  filtering, and a broken cache object must not raise). 98 passed, 3 skipped.

Docs (parameter-and-kv-management-status.md):
- new 2.5: three sources for the progress reading, and why the engine-side one
  is not enough on its own
- new 2.6: what a replacement machine actually needs after an interruption.
  Key point: an activation only advances the current step; continuing needs the
  accumulated KV, so "move the activation" is not sufficient. Weights are a
  static asset and never need migrating. Quantified transfer-vs-recompute:
  moving KV costs the same bytes as re-running the history but blocks for ~4x
  less and needs no cooperation from upstream; the first stage is the exception
  (it has no upstream and can recompute itself more cheaply).
- new 10.1: measured per-layer bytes (466.1 MB, read from safetensors headers)
  and the mapping from layer ranges to HF shard files. HF shard boundaries do
  not align with layer boundaries: for a 12/4/4/8 split the stages need shards
  {1,2} {2,3} {3} {3,4}, so no machine needs all 15.2 GB. Recommends repacking
  into one file per stage (6.7 / 1.9 / 1.9 / 4.8 GB), which also speeds up
  prepare_epoch given the measured 110 MiB/s DrvFs read.

Scripts:
- scripts/probe_7b_split_map.py (new): per-layer bytes and per-stage shard
  requirements, parsed from safetensors headers without loading tensors
- scripts/probe_kv_accounting.py: also prints request_positions()
EOF
)

git commit -m "$MSG"
echo
git log --oneline -3
echo
echo "=== push ==="
git push origin main
git status -sb | head -2
