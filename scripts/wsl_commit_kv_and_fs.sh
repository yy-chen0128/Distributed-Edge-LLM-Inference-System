#!/usr/bin/env bash
# Commit the KV-accounting fix, fs benchmark, download fix and doc updates.
# ASCII ONLY (PowerShell pipe re-encodes).
set -eu
cd "$HOME/pair"

echo "=== status ==="
git status --short
git add -A
echo
echo "=== staged ==="
git diff --cached --stat

MSG=$(cat <<'EOF'
fix: KV accounting returned 0; measure filesystem read and GPU memory

kv_bytes was broken: it read `cache[i]`, but transformers 5.x DynamicCache is
not subscriptable, so the TypeError was swallowed by `except ... continue` and
the function always returned 0 -- that is why every stage in the 4-stage run
reported kv_bytes_last = 0. Now it walks cache.layers[i].keys/.values with a
fallback for 4.4x key_cache/value_cache. Measured after the fix:
  6 layers, 38 tokens  -> 116,736 B (exactly 2*2*64*2*6*38)
  +5 decode steps      -> 132,096 B (+3072 per step)
  after release        -> 0
Also: status() now reports per-request KV bytes.
Also documented: torch.cuda.memory_allocated delta is NOT a KV proxy (measured
8.64 MB for 0.117 MB of KV, 74x, and it does not drop on release).

hf_mirror_download.py: hf-mirror returns 403 for the default Python-urllib
User-Agent; now sends a browser UA (verified: default 403, browser UA 200).
Unblocks the Qwen2.5-7B-Instruct download.

New measurements:
- scripts/measure_fs_read.py: cold vs warm file reads using
  posix_fadvise(DONTNEED). /mnt/d (DrvFs) = 112.7 cold / 114.5 warm MiB/s,
  ext4 = 2319 MiB/s -> 20x. DrvFs caching does not help, so every epoch switch
  pays full price. This explains prepare_epoch (451 MB / 110 MiB/s ~ 4.1 s,
  measured 3.3-8.2 s) and implies ~57 s for a 12-layer 7B shard.
- scripts/probe_kv_accounting.py, scripts/probe_kv_api.py: KV/API probe
- tests/test_kv_accounting.py: regression tests without a real model

Docs:
- single-machine-scope-and-batching.md: new sections on GPU parameter/KV state
  (what is implemented, what is missing, measured numbers) and on /mnt vs ext4
- real-workload-datasets.md: explicit exclusion decisions, clarification that
  the missing churn data is a data gap (not a dataset-selection issue), and a
  new section on model size / quantization (incl. why AWQ/GPTQ would silently
  produce garbage in the current layered engine)
- edge-4gpu-deployment-analysis.md: prepare cost now explained by the measured
  DrvFs read throughput
EOF
)

git commit -m "$MSG"
echo
git log --oneline -3
echo
echo "=== push ==="
git push origin main
git status -sb | head -2
