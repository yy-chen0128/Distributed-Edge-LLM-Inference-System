#!/usr/bin/env bash
# Fix the two issues found by V1a:
#   1. transformers 5.x breaks vLLM 0.10.1 (it needs 4.x) -> pin <5
#   2. LMCache's c_ops "libc10.so" error was an artefact of importing lmcache
#      BEFORE torch; re-test with torch first.
# Then restart the server and run the first real request.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"
PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct
PORT=8000
LOG=.models/logs/vllm_v1b_server.log
export PIP_CACHE_DIR=/mnt/d/pipcache
IDX=https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== current versions ==="
"$PY" -c "import transformers, tokenizers; print('transformers', transformers.__version__, '| tokenizers', tokenizers.__version__)"

echo
echo "=== pin transformers to 4.x (vLLM 0.10.1 is written against 4.x) ==="
"$VENV/bin/pip" install -i "$IDX" --timeout 60 --retries 5 \
  "transformers>=4.53.2,<5" "tokenizers<0.22" 2>&1 | tail -6

echo
echo "=== versions after ==="
"$PY" - <<'EOF' 2>&1 | tail -14
import torch                     # FIRST: loads libc10.so etc into the process
print("  torch", torch.__version__, "cuda", torch.cuda.is_available())
import transformers, tokenizers
print("  transformers", transformers.__version__, "| tokenizers", tokenizers.__version__)
print("  has all_special_tokens_extended:",
      hasattr(transformers.PreTrainedTokenizerBase, "all_special_tokens_extended"))
try:
    import lmcache
    print("  lmcache import OK:", lmcache.__file__.split("site-packages/")[-1])
except Exception as exc:
    print("  lmcache FAILED:", type(exc).__name__, str(exc)[:120])
try:
    import lmcache.c_ops as c
    print("  lmcache.c_ops OK (torch imported first)")
except Exception as exc:
    print("  lmcache.c_ops FAILED:", type(exc).__name__, str(exc)[:160])
try:
    import vllm
    print("  vllm", vllm.__version__)
except Exception as exc:
    print("  vllm FAILED:", type(exc).__name__, str(exc)[:120])
EOF

echo
echo "=== start vLLM again ==="
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
  --max-model-len 8192 --gpu-memory-utilization 0.55 \
  > "$LOG" 2>&1 &
SPID=$!
ready=0
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" != "1" ]; then
  echo "STILL NOT READY after $((i*5))s; last 20 lines:"
  tail -20 "$LOG"
  kill "$SPID" 2>/dev/null
  exit 1
fi
echo "READY after ~$((i*5))s"
grep -iE 'Starting vLLM|vLLM API server|max_model_len|Capturing|init engine|Available KV' "$LOG" | head -8

echo
echo "=== first real request (3 requests sharing a 3000-token prefix) ==="
"$PY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$PORT" \
  --prefix-tokens 3000 --requests 3 --max-tokens 16 --label V1b-plain 2>&1 | tail -8

echo
echo "=== stop ==="
kill "$SPID" 2>/dev/null
sleep 4
pkill -f 'vllm serve' 2>/dev/null
echo done
