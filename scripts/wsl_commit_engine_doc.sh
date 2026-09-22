#!/usr/bin/env bash
# Commit: engine selection doc, measured 7B stage numbers, clone/verify scripts.
# ASCII ONLY.
set -eu
cd "$HOME/pair"

echo "=== sanity: third-party clones must NOT be tracked ==="
git status --short | grep -E '(ollama|lmstudio|lms)/' && { echo "ABORT: clone staged"; exit 1; } || echo "ok: no clone in status"

git add -A
echo
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: engine selection analysis; measured 7B stage replaces the projection

Engine selection (docs/research/inference-engine-selection.md):
- six requirements (R1..R6) and nine candidate routes
- Ollama: verified against the CLONED source (envconfig/config.go, server/sched.go,
  llm/llama_server.go, api/types.go). No cross-machine mechanism anywhere; the
  only remote-looking variables (OLLAMA_REMOTES / OLLAMA_CREATE_REMOTE) are about
  where models are fetched from, not remote compute; layer count is passed to the
  llama-server child process as -ngl and the user cannot set it per device.
- LM Studio: LM Link is whole-model remote serving over Tailscale, not a split;
  server is closed source and the EULA forbids modification/derivation/redistribution,
  so it can only be a black-box baseline.
- llama.cpp RPC backend: official README says weights AND KV are distributed across
  local and remote devices in proportion to available memory (--tensor-split to
  override), and labels itself "proof-of-concept ... fragile and insecure".
  Cheapest cross-machine baseline, and it brings GGUF quantisation for free.
- exo: does pipeline sharding with topology-aware auto-parallel and a placement
  preview API, but is MLX/Apple-centric and runs on CPU only on Linux -> unusable
  on our Windows/NVIDIA laptops; valuable as a reference design.
- vLLM/SGLang: homogeneous and static PP, no elasticity.
- Recommendation: keep the execution plane in-house (only route satisfying R1+R2),
  borrow single-machine capabilities (GGUF quantisation, slot/Unified-KV batching,
  checkpointed prefix cache, metric definitions), and build THREE baselines
  (single-machine Ollama, cross-machine llama.cpp RPC, datacenter vLLM).

Measured 7B stage (scripts/probe_7b_stage.py) replaces the earlier projection:
- per layer: 466.1 MB resident, 2.55 ms fixed per call, 0.0217 ms/token marginal,
  2.60 ms/packet decode; KV 2048 B/token/layer at the measured position
- the earlier projection assumed c0 scales with weight size. It does not: c0 grew
  only 2.6x while c1 grew 10.1x, so the 7B overhead share at 38 tokens is 75.6%
  (not the projected 43.5%) and the break-even prompt length is 118 tokens
  (not 27). Docs corrected.
- prepare_epoch runs at ~53 MB/s effective, not the 110 MiB/s raw read: 7B four
  layers (1864.5 MB) took 35.7 s, first four layers with the 1.09 GB embedding
  55.5 s. A 12-layer stage (~6.7 GB) is therefore ~126 s per reconfiguration,
  not the 52 s implied by raw read speed.

.gitignore: clones of ollama/, lmstudio-python/, lms/ are read-only references and
must never be tracked (same rule as vllm/ and LMCache/).

Scripts: wsl_clone_engines*.sh, wsl_verify_engines.sh (source-level checks),
probe_7b_stage.py (real 7B stage measurement).
EOF
git commit -F "$MSGFILE"
rm -f "$MSGFILE"

echo
git log --oneline -3
echo
echo "=== push ==="
git push origin main
git status -sb | head -2
echo "--- tracked file count ---"
git ls-files | wc -l
