#!/usr/bin/env bash
# Verify the Ollama/LM Studio claims against the cloned source. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== repo state ==="
for d in ollama lmstudio-python lms; do
  printf '%-18s ' "$d"
  git -C "$d" log -1 --format='%h %cd %s' --date=short 2>&1 | head -1
done

echo
echo "=== [A] Ollama: is there ANY cross-machine knob? ==="
echo "--- envconfig: every variable mentioning dist/rpc/remote/cluster ---"
grep -rn -iE '"(OLLAMA|GGML)_[A-Z_]*(RPC|DIST|REMOTE|CLUSTER|WORKER)' ollama/envconfig/ || echo "  (none)"
echo "--- files under envconfig ---"
ls ollama/envconfig/
echo "--- 'rpc' anywhere in the Go source (excluding vendor/tests) ---"
grep -rn -i 'rpc' --include='*.go' ollama/ | grep -v '_test.go' | head -20 || echo "  (none)"

echo
echo "=== [B] Ollama: how are layers split across GPUs (and can a user set it)? ==="
echo "--- NumGPU / layer offload references ---"
grep -rn 'NumGPU' --include='*.go' ollama/ | grep -v '_test.go' | head -15
echo "--- scheduling sources ---"
ls ollama/server/ 2>/dev/null | head -20
echo "--- does the API surface have any per-request device/placement field? ---"
grep -rn -iE 'num_gpu|device|placement' ollama/api/types.go | head -10 || echo "  (none in api/types.go)"

echo
echo "=== [C] Ollama: batch / parallelism knobs ==="
grep -rn -E 'NumParallel|MaxQueue|MaxLoadedModels|SchedSpread' ollama/envconfig/config.go | head -10

echo
echo "=== [D] LM Studio SDKs: client-only? ==="
echo "--- lmstudio-python top level ---"
ls lmstudio-python/ | head -15
echo "--- any server/engine/scheduler implementation in the SDK? ---"
grep -rln -iE 'class .*Scheduler|paged_attention|kv_connector|layer_split' lmstudio-python/ lms/ 2>/dev/null | head -5 || echo "  (none - SDK is a client)"
echo "--- what transport does it use? ---"
grep -rn -iE 'http|websocket|localhost|1234' lmstudio-python/README.md 2>/dev/null | head -5
