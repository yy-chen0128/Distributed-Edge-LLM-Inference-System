#!/usr/bin/env bash
# V1c: get the real 400 error, and hunt for an LMCache wheel whose C extension
# matches torch 2.7.1. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"
PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct
PORT=8000
LOG=.models/logs/vllm_v1c_server.log
export PIP_CACHE_DIR=/mnt/d/pipcache
IDX=https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== start server ==="
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
  --max-model-len 8192 --gpu-memory-utilization 0.55 > "$LOG" 2>&1 &
SPID=$!
ready=0
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 5
done
[ "$ready" != "1" ] && { echo "NOT READY"; tail -15 "$LOG"; kill "$SPID" 2>/dev/null; exit 1; }
echo "ready after ~$((i*5))s"
echo "--- what /v1/models says ---"
curl -s "http://127.0.0.1:$PORT/v1/models"
echo
echo
echo "=== minimal request, printing any error body ==="
"$PY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$PORT" \
  --prefix-tokens 50 --requests 1 --max-tokens 8 --label minimal 2>&1 | tail -12

echo
echo "=== raw curl against /v1/completions (non-streaming) ==="
curl -s -X POST "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "$($PY -c "
import json,urllib.request
m=json.load(urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models'))['data'][0]['id']
print(json.dumps({'model':m,'prompt':'hello','max_tokens':8,'temperature':0.0}))")" | head -c 500
echo
echo
kill "$SPID" 2>/dev/null; sleep 4; pkill -f 'vllm serve' 2>/dev/null

echo
echo "=== LMCache wheel hunt (looking for one that imports with torch 2.7.1) ==="
for V in 0.3.6 0.3.5 0.3.3 0.3.1; do
  echo "--- lmcache==$V ---"
  "$VENV/bin/pip" install -q -i "$IDX" --timeout 60 --retries 3 \
      "torch==2.7.1" "lmcache==$V" 2>&1 | tail -2
  "$PY" - <<'EOF' 2>&1 | tail -3
import torch
try:
    import lmcache.c_ops
    print("   c_ops OK")
except Exception as exc:
    print("   c_ops FAILED:", type(exc).__name__, str(exc)[:110])
EOF
done
