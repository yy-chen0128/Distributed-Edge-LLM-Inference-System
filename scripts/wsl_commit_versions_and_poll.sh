#!/usr/bin/env bash
# Commit the working version combination + poll the four-phase smoke test.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/v1_smoke

git add -A
git diff --cached --stat
MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs+fix: the working vLLM/LMCache version set for a CUDA-12.7 driver

Decision taken: do not upgrade the driver; use the older combination. Measured
working set on driver 566.24 (CUDA 12.7):
  vllm 0.10.1 + torch 2.7.1+cu126 + ray 2.58.0 + lmcache 0.3.5
with two non-obvious pins discovered the hard way:
- transformers MUST be <5. vLLM 0.10.1 declares transformers>=4.53.2, so pip
  installs 5.x and the server then dies at startup with
  "Qwen2Tokenizer has no attribute all_special_tokens_extended". Pinned to 4.55.4.
- lmcache MUST be <=0.3.5. From 0.3.6 the wheel's C extension has an ABI mismatch
  with torch 2.7.1: "undefined symbol: _ZN3c104cuda9SetDeviceEab"
  (c10::cuda::SetDevice). 0.3.5 and 0.3.3 import cleanly, and
  lmcache.integration.vllm.lmcache_connector_v1 imports, so vLLM 0.10.1 can use it.

Also records the measured startup cost, which is the first component of the tier-0
bubble: vllm serve reaches ready in 85-105 s for the 0.5B model, of which 34.5 s is
engine init (profile, KV cache, warmup) and 3 s is CUDA graph capture; available KV
memory is 2.07 GiB at gpu-memory-utilization 0.55.

And a methodology fix that mattered: the bench client built its "3000-token"
prefix out of ctx00000-style words, which tokenise into several tokens each, so the
real prompt exceeded max-model-len 8192 and the server correctly returned HTTP 400.
The client now uses " the" (about one token per word), reports the real prompt_tokens
from the response usage, and prints the server's error body instead of just
"HTTP Error 400". Lesson recorded in the doc: any length experiment must report the
tokenizer-measured token count, never an estimate from word counts.
EOF
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -F "$MSGFILE" >/dev/null
rm -f "$MSGFILE"
echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -2

echo
echo "=== smoke test progress ==="
if [ -f .models/logs/v1_smoke.pid ] && kill -0 "$(cat .models/logs/v1_smoke.pid)" 2>/dev/null; then
  echo "smoke ALIVE (pid $(cat .models/logs/v1_smoke.pid))"
else
  echo "smoke: finished or gone"
fi
echo "--- phase markers so far ---"
grep -E '^#####|^  \[|ready|NOT READY' "$OUT/driver.log" 2>/dev/null | tail -12
echo
echo "--- bench results so far ---"
for f in "$OUT"/bench_*.log; do
  [ -f "$f" ] || continue
  printf '%-24s %s\n' "$(basename "$f" .log)" "$(grep -h 'JSON' "$f" | tail -1)"
done
echo
echo "--- driver tail ---"
tail -8 "$OUT/driver.log" 2>/dev/null
