#!/usr/bin/env bash
# Pin the working LMCache (0.3.5), verify it, then run the FOUR-PHASE cache test:
#   A plain vLLM warm | B plain vLLM after restart | C LMCache warm | D LMCache after restart
# D is the number that decides the tier-1 (replay) bubble size.
# Runs in the background; poll .models/logs/v1_smoke_*.log
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"
PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct
PORT=8000
OUT=.models/logs/v1_smoke
export PIP_CACHE_DIR=/mnt/d/pipcache
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p "$OUT"

cat > .models/logs/run_v1_smoke.sh <<'OUTER'
#!/usr/bin/env bash
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"; PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct; PORT=8000; OUT=.models/logs/v1_smoke
BASE="http://127.0.0.1:$PORT"
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_CACHE_DIR=/mnt/d/pipcache
mkdir -p "$OUT"

cat > "$OUT/lmcache.yaml" <<'YAML'
chunk_size: 256
local_cpu: true
max_local_cpu_size: 4
YAML

start() {  # start <tag> [extra args...]
  local tag="$1"; shift
  local log="$OUT/server_$tag.log"
  nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
      --max-model-len 8192 --gpu-memory-utilization 0.55 "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && { echo "  [$tag] ready"; return 0; }
    sleep 5
  done
  echo "  [$tag] NOT READY"; tail -15 "$log"; return 1
}
stop() {
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
  sleep 6; pkill -f 'vllm serve' 2>/dev/null; sleep 4
}
bench() {  # bench <label>
  "$PY" scripts/vllm_bench_prefix.py --base-url "$BASE" \
      --prefix-tokens 3000 --requests 3 --max-tokens 16 --label "$1" 2>&1 \
      | tee "$OUT/bench_$1.log"
}

echo "=== pin lmcache 0.3.5 (0.3.6's c_ops has an ABI mismatch with torch 2.7.1) ==="
"$VENV/bin/pip" install -q -i "$IDX" --timeout 60 --retries 3 "torch==2.7.1" "lmcache==0.3.5" 2>&1 | tail -2
"$PY" - <<'EOF'
import torch
import lmcache
from lmcache._version import __version__ as v
print("  lmcache", v)
try:
    import lmcache.c_ops
    print("  c_ops OK")
except Exception as exc:
    print("  c_ops FAILED:", type(exc).__name__, str(exc)[:110])
for mod in ("lmcache.integration.vllm.lmcache_connector_v1",
            "lmcache.integration.vllm.lmcache_connector"):
    try:
        __import__(mod); print(f"  {mod}: OK")
    except Exception as exc:
        print(f"  {mod}: {type(exc).__name__}")
EOF

echo
echo "########## A: plain vLLM, 3 requests, same 3000-token prefix ##########"
start A-plain && bench A-plain-warm
echo
echo "########## B: plain vLLM restarted -> in-process cache should be gone ##########"
stop
start B-plain2 && bench B-plain-restart
stop

KVCFG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
export LMCACHE_CONFIG_FILE="$HOME/pair/$OUT/lmcache.yaml"
echo
echo "########## C: vLLM + LMCache (0.3.5) ##########"
if start C-lmcache --kv-transfer-config "$KVCFG"; then
  bench C-lmcache-warm
  echo
  echo "########## D: vLLM + LMCache RESTARTED -> does the cache survive? ##########"
  stop
  if start D-lmcache2 --kv-transfer-config "$KVCFG"; then bench D-lmcache-restart; fi
else
  echo "LMCache connector failed; see $OUT/server_C-lmcache.log"
fi
stop

echo
echo "=== SUMMARY ==="
for f in "$OUT"/bench_*.log; do
  [ -f "$f" ] || continue
  printf '%-24s %s\n' "$(basename "$f" .log)" "$(grep -h 'JSON' "$f" | tail -1)"
done
echo "=== channels ==="
for f in "$OUT"/server_*.log; do
  [ -f "$f" ] || continue
  printf '%-26s ' "$(basename "$f" .log)"
  grep -c -iE 'LMCache|kv_connector|cache' "$f" | sed 's/^/lmcache-log-lines=/'
done
echo "### V1_SMOKE_DONE"
OUTER

nohup bash .models/logs/run_v1_smoke.sh > "$OUT/driver.log" 2>&1 &
echo $! > .models/logs/v1_smoke.pid
echo "V1 four-phase smoke started: pid $(cat .models/logs/v1_smoke.pid)"
echo "logs: $OUT/driver.log  (poll it)"
sleep 40
echo
echo "--- driver.log so far ---"
tail -20 "$OUT/driver.log"
