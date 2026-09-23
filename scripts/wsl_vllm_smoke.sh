#!/usr/bin/env bash
# V1 smoke test: does vLLM run, and does LMCache give us a prefix cache that
# SURVIVES a server restart (the thing that matters for tier-0 restart elasticity)?
#
# Four measurements:
#   A  plain vLLM, warm requests            -> built-in prefix cache is warm
#   B  plain vLLM, restarted, same prefix   -> cold again (in-process cache is gone)
#   C  vLLM + LMCache, warm requests        -> warm
#   D  vLLM + LMCache, restarted, same prefix -> warm IF LMCache persisted
#
# ASCII ONLY.
set -u
VENV="${VENV:-$HOME/venvs/vllm}"
MODEL="${MODEL:-.models/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:$PORT"
PREFIX_WORDS="${PREFIX_WORDS:-3500}"
REQS="${REQS:-3}"
LOGDIR=".models/logs"
mkdir -p "$LOGDIR"

cd "$HOME/pair" || exit 1
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "NO_VLLM_VENV: run scripts/wsl_install_vllm.sh first"; exit 1; }
[ -d "$MODEL" ] || { echo "NO_MODEL: $MODEL"; exit 1; }

# LMCache local CPU cache config (minimal)
LMCACHE_CFG="$HOME/pair/.models/logs/lmcache.yaml"
cat > "$LMCACHE_CFG" <<'YAML'
chunk_size: 256
local_cpu: true
max_local_cpu_size: 4
YAML

wait_ready() {
  for _ in $(seq 1 120); do
    if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}

start_server() {
  local tag="$1"; shift
  local log="$LOGDIR/vllm_$tag.log"
  echo "--- starting vLLM [$tag] $* ---"
  nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
    --max-model-len 8192 --gpu-memory-utilization 0.55 "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  if wait_ready; then
    echo "--- [$tag] ready (pid $SERVER_PID) ---"
    return 0
  fi
  echo "--- [$tag] NOT READY; last log lines: ---"
  tail -n 25 "$log"
  kill "$SERVER_PID" 2>/dev/null
  return 1
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 5
  fi
  pkill -f "vllm serve" 2>/dev/null
  sleep 3
}

bench() {
  "$PY" scripts/vllm_bench_prefix.py --base-url "$BASE" \
    --prefix-tokens "$PREFIX_WORDS" --requests "$REQS" --label "$1" \
    2>&1 | tee "$LOGDIR/bench_$1.log"
}

echo "=== model=$MODEL venv=$VENV port=$PORT prefix_words=$PREFIX_WORDS ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

echo
echo "########## A: plain vLLM, warm ##########"
if start_server plain; then
  bench A-plain-warm
  echo "########## B: plain vLLM after RESTART (in-process cache should be gone) ##########"
  stop_server
  if start_server plain2; then bench B-plain-restart; fi
fi
stop_server

echo
echo "########## C/D: vLLM + LMCache ##########"
KVCFG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
export LMCACHE_CONFIG_FILE="$LMCACHE_CFG"
if start_server lmcache --kv-transfer-config "$KVCFG"; then
  bench C-lmcache-warm
  echo "########## D: LMCache after RESTART (should still be warm if it persisted) ##########"
  stop_server
  if start_server lmcache2 --kv-transfer-config "$KVCFG"; then bench D-lmcache-restart; fi
else
  echo "!! LMCache connector failed; try the other connector name:"
  echo "   -kv-transfer-config '{\"kv_connector\":\"LMCacheConnector\",\"kv_role\":\"kv_both\"}'"
fi
stop_server

echo
echo "=== summary: TTFT lists per phase (cold first request vs warm rest) ==="
for f in "$LOGDIR"/bench_*.log; do
  [ -f "$f" ] || continue
  printf '%-28s %s\n' "$(basename "$f" .log)" "$(grep -h 'JSON \[' "$f" | tail -1)"
done
echo "=== done ==="
