#!/usr/bin/env bash
# Commit the concept/status doc and the corrected 7B numbers. ASCII ONLY.
set -eu
cd "$HOME/pair"

git add -A
git diff --cached --stat

MSG=$(cat <<'EOF'
docs: parameter/KV management status (written for non-code readers)

Adds docs/research/parameter-and-kv-management-status.md. Every concept is
defined first (token, prefill/decode, KV cache, layer, vocabulary/embedding/
lm_head, progress_tokens), then mapped to its symbol in the code, then marked
as implemented / fields-only / missing. Contents:

- progress_tokens: what it is (a per-request progress counter), why it matters
  (saves ~11 s of rework for a 7.6K prompt + 150 generated tokens at 7B), and
  why it is dead today. The bug is two holes, not one: (1) nothing in the
  framework ever writes the field, because the autoregressive loop lives in
  run_real_pipeline.py while the recovery policy lives in TaskScheduler, and
  handle_request advances exactly one token per call; (2) even with the field
  set, the resuming node does not have the prefix KV (that is the missing
  export/import capability). The dangerous half of hole 2 was fixed earlier
  (no cache -> recompute the whole prompt instead of feeding one token).
- Parameter management: meta-device skeleton + per-stage load is implemented and
  measured; offload / quantisation / cross-process sharing are not.
- KV management: per-request lifetime and accounting work (accounting was fixed
  in the previous commit); "hit" currently affects routing only, never compute;
  migration exists only at the block-abstraction layer with no bridge to the
  real tensors, and the HF engine returns empty kv_block_ids so the store is
  always empty in real runs.
- Paging: not needed for our goals, and do not reimplement it; what we need is
  layer-addressable export/import. Notes the vLLM trade-off (paging + KV
  connector for free, but vLLM PP is homogeneous and static).
- MoE expert offload: fields exist, no code; recommendation is to skip it.
- Vocabulary splitting: retracted as a plan -- it is tensor parallelism and
  conflicts with the measured "no TP over WiFi" result; the correct fix is to
  treat the vocabulary as a fixed cost when choosing the split.
- 7B on 4 laptops: per-layer 0.466 GB (GQA: 4 kv heads), vocabulary 1.09 GB per
  copy, KV 56 KB/token. Equal split does NOT fit 8/6/4/4 (stages 2 and 3
  overflow); weighted 12/4/4/8 and 8/6/6/8 do. Flags that
  CapabilityReparallelization has no VRAM constraint and would produce an
  unloadable split.

Corrections:
- 7B per-layer weight 510 MB -> 466 MB (the old figure ignored GQA), so the
  0.5B->7B marginal ratio is 15.6x (was 17.1x) and the break-even prompt length
  for 7B is ~30 tokens (was 27). Table in single-machine-scope-and-batching.md
  updated.
- 7B vocabulary matrix 0.54 GB -> 1.09 GB per copy (152064 x 3584 x 2 B), and
  since tie_word_embeddings is false there are two copies (2.18 GB total).
- scripts/measure_stage_scaling.py: projection formula now includes GQA.
- scripts/model_pp_fit.py: same User-Agent fix as the downloader (the mirror
  returns 403 for the default Python-urllib UA).
EOF
)

git commit -m "$MSG"
echo
git log --oneline -3
echo
echo "=== push ==="
git push origin main
git status -sb | head -2
