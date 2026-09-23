#!/usr/bin/env bash
# V2b retry: local Ray + PP=4. The first attempt failed because VLLM_HOST_IP was
# set to 127.0.0.1 while the Ray node lives on the LAN IP, so the placement group's
# node-affinity label matched nothing. This time: do not force VLLM_HOST_IP; let
# vLLM use the Ray node's own address.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"; PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct; PORT=8020
OUT=.models/logs/v2b; mkdir -p "$OUT"
export PYTHONPATH=.

echo "=== this machine's addresses (note: campus DHCP, it changes) ==="
ip -4 addr show 2>/dev/null | grep 'inet ' | sed 's/^/  /'
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "  picked LOCAL_IP=$LOCAL_IP"

echo
echo "=== reset ray ==="
pkill -f 'vllm serve' 2>/dev/null || true
"$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 3

echo "=== ray start --head --num-gpus=4 (all four virtual GPUs on the one real GPU) ==="
"$VENV/bin/ray" start --head --num-gpus=4 --port=6379 \
   --min-worker-port=10002 --max-worker-port=10100 > "$OUT/ray_start2.log" 2>&1
sleep 8
"$VENV/bin/ray" status 2>&1 | sed -n '1,25p'
echo "--- resources ---"
"$VENV/bin/ray" status --format=json 2>/dev/null | "$PY" -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print('  cluster resources:', d.get('clusterResources'))
except Exception as exc:
    print('  (could not parse)', exc)
"

echo
echo "=== vllm serve PP=4 via ray (NO VLLM_HOST_IP override) ==="
unset VLLM_HOST_IP MASTER_ADDR MASTER_PORT 2>/dev/null || true
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
   --pipeline-parallel-size 4 --distributed-executor-backend ray \
   --max-model-len 4096 --gpu-memory-utilization 0.30 \
   > "$OUT/server2.log" 2>&1 &
SPID=$!
ready=0
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 5
done
if [ "$ready" = "1" ]; then
  echo "PP=4 (ray) READY after ~$((i*5))s"
  grep -iE 'placement group|pipeline|rank|PP group|world|init engine|Starting vLLM' "$OUT/server2.log" | head -12
  echo
  "$PY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$PORT" \
      --prefix-tokens 500 --requests 2 --max-tokens 16 --label PP4-ray 2>&1 | tail -6
else
  echo "NOT READY after $((i*5))s; last 20 lines:"
  tail -20 "$OUT/server2.log"
fi

echo
echo "=== cleanup ==="
kill "$SPID" 2>/dev/null; sleep 3
pkill -f 'vllm serve' 2>/dev/null || true
"$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
echo "### V2B2_DONE"
