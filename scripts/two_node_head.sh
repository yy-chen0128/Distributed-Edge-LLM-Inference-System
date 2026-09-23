#!/usr/bin/env bash
# Run on the HEAD machine (A). Two-machine bring-up plus the decisive measurement:
#
#   WORKER_IP=<B's LAN IP> bash scripts/two_node_head.sh
#
# Phase 1  PP=1 on this machine alone            -> baseline tokens/s and TPOT
# Phase 2  PP=2 across A and B over the network  -> same measurements
# then it prints the difference, which IS the per-token cost of crossing machines.
# If the difference is large relative to TPOT, a single token traversing two
# machines is network-bound and the design needs rethinking (that is the whole
# point of measuring before building).
#
# Optional: WORKER_SSH="user@ip" (with -p via WORKER_SSH_PORT) lets this script start
# the worker Ray itself; otherwise you run two_node_worker.sh on B by hand.
#
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="${VENV:-$HOME/venvs/vllm}"
PAIRPY="${PAIRPY:-$HOME/venvs/pair/bin/python}"
# NOTE: no apostrophes inside ${VAR:?...} -- bash mis-parses them.
WORKER_IP="${WORKER_IP:?set WORKER_IP to the LAN IP of the worker machine}"
WORKER_SSH="${WORKER_SSH:-}"
WORKER_SSH_PORT="${WORKER_SSH_PORT:-22}"
WORKER_REPO="${WORKER_REPO:-~/pair}"
MODEL="${MODEL:-.models/Qwen2.5-0.5B-Instruct}"
PROMPT_WORDS="${PROMPT_WORDS:-1000}"
MAX_TOKENS="${MAX_TOKENS:-64}"
VPORT="${VPORT:-8000}"
GPORT="${GPORT:-8100}"
OUT=.models/logs/two_node; mkdir -p "$OUT"

ROUTE="$(ip route get "$WORKER_IP" 2>/dev/null | head -1)"
IFACE="$(echo "$ROUTE" | grep -oP 'dev \K[^ ]+' | head -1)"
SRCIP="$(echo "$ROUTE" | grep -oP 'src \K[0-9.]+' | head -1)"
if [ -z "$IFACE" ]; then echo "cannot route to $WORKER_IP -> fix networking first"; exit 1; fi
export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"
export VLLM_HOST_IP="$SRCIP"

echo "================= PREFLIGHT ================="
echo "  head   : $(hostname)  LAN $SRCIP (iface $IFACE)"
echo "  worker : $WORKER_IP"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | sed 's/^/  head gpu: /'
if ping -c 3 -W 2 "$WORKER_IP" >/dev/null 2>&1; then
  echo "  ping worker: ok"
else
  echo "  ping worker: FAILED -> campus WiFi client isolation? fix before continuing"
fi
echo "  rtt sample:"; ping -c 5 -q "$WORKER_IP" 2>/dev/null | tail -2 | sed 's/^/    /'

stop_all() {
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f vllm_gateway 2>/dev/null || true
  "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
  sleep 4
}

serve_wait() {  # serve_wait <log> <extra args...>
  local log="$1"; shift
  nohup "$VENV/bin/vllm" serve "$MODEL" --port "$VPORT" "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$VPORT/v1/models" >/dev/null 2>&1 && { echo "  ready after ~$((i*5))s"; return 0; }
    sleep 5
  done
  echo "  NOT READY; last 15 lines:"; tail -15 "$log"; return 1
}

bench() {
  "$PAIRPY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$VPORT" \
      --prefix-tokens "$PROMPT_WORDS" --requests 2 --max-tokens "$MAX_TOKENS" \
      --ignore-eos --label "$1" 2>&1 | tee "$OUT/bench_$1.log"
}

echo
echo "================= PHASE 1: PP=1 on this machine (baseline) ================="
stop_all
if serve_wait "$OUT/server_pp1.log" --max-model-len 4096 --gpu-memory-utilization 0.55; then
  bench PP1-single-machine
fi
stop_all

echo
echo "================= PHASE 2: PP=2 across two machines (Ray) ================="
"$VENV/bin/ray" start --head --port=6379 --num-gpus=1 \
   --min-worker-port=10002 --max-worker-port=10100 > "$OUT/ray_head.log" 2>&1
sleep 6
ADDR_LINE="$(grep -o 'ray start --address=[^ ]*' "$OUT/ray_head.log" | head -1)"
echo "  ray head up: ${ADDR_LINE:-see $OUT/ray_head.log}"

if [ -n "$WORKER_SSH" ]; then
  echo "  starting the worker over ssh ($WORKER_SSH -p $WORKER_SSH_PORT)..."
  ssh -p "$WORKER_SSH_PORT" -o BatchMode=yes "$WORKER_SSH" \
      "cd $WORKER_REPO && HEAD_IP=$SRCIP MODEL=$MODEL nohup bash scripts/two_node_worker.sh > .models/logs/two_node/worker.log 2>&1 &" \
      && echo "  worker launch requested"
else
  echo "  >>> NOW run this on the worker machine (B):"
  echo "        cd <repo> && HEAD_IP=$SRCIP bash scripts/two_node_worker.sh"
fi

echo "  waiting for 2 nodes to join..."
NODES=0
for i in $(seq 1 60); do
  NODES=$("$VENV/bin/ray" status 2>/dev/null | awk '/^Active:/{f=1;next} /^Pending:/{f=0} f' | grep -c 'node_')
  [ "$NODES" -ge 2 ] && { echo "  $NODES nodes joined after ~$((i*5))s"; break; }
  sleep 5
done
"$VENV/bin/ray" status 2>&1 | sed -n '1,14p' | sed 's/^/    /'
if [ "$NODES" -lt 2 ]; then
  echo "  only $NODES node(s) -> PP=2 cannot start; fix the worker join first"
  exit 1
fi

if serve_wait "$OUT/server_pp2.log" --pipeline-parallel-size 2 \
      --distributed-executor-backend ray --max-model-len 4096 \
      --gpu-memory-utilization 0.45; then
  echo "  --- evidence that the pipeline spans two machines: ---"
  grep -iE 'pipeline|PP group|rank|world_size|Starting vLLM|Available KV' "$OUT/server_pp2.log" | head -10 | sed 's/^/    /'
  bench PP2-two-machines
fi

echo
echo "================= RESULT ================="
"$PAIRPY" - "$OUT" <<'EOF'
import json, os, sys, re
out = sys.argv[1]
def grab(name):
    p = os.path.join(out, f"bench_{name}.log")
    if not os.path.exists(p):
        return None
    for line in open(p, encoding="utf-8", errors="replace"):
        if "] JSON " in line:
            try:
                return json.loads(line.split("] JSON ", 1)[1])
            except ValueError:
                pass
    return None
pp1, pp2 = grab("PP1-single-machine"), grab("PP2-two-machines")
print(f"  {'config':<24}{'warm TPOT ms':>14}{'tok/s':>9}{'warm TTFT ms':>14}")
for name, d in (("PP=1 single machine", pp1), ("PP=2 two machines", pp2)):
    if d:
        print(f"  {name:<24}{d['warm_tpot_ms']:>14.2f}{d['tps'][-1]:>9.1f}{d['ttft_ms'][-1]:>14.1f}")
    else:
        print(f"  {name:<24}{'(no data)':>14}")
if pp1 and pp2:
    delta = pp2["warm_tpot_ms"] - pp1["warm_tpot_ms"]
    ratio = pp2["warm_tpot_ms"] / max(1e-9, pp1["warm_tpot_ms"])
    print()
    print(f"  per-token cost added by crossing machines: {delta:+.2f} ms  ({ratio:.2f}x)")
    print("  Interpretation: this is the number that decides whether multi-machine PP")
    print("  is worth it on THIS network. Compare it against the per-token compute")
    print("  time; if the delta dominates, the design must change (fewer hops, larger")
    print("  batches, or a different split).")
EOF

echo
echo "  logs under $OUT/"
echo "  to also run the tier-0/tier-1 drill against this two-machine setup:"
echo "    PYTHONPATH=. nohup $PAIRPY -m edge_llm_scheduler.gateway.vllm_gateway \\"
echo "        --upstream http://127.0.0.1:$VPORT --port $GPORT --stall-timeout 10 &"
echo "    $PAIRPY scripts/gateway_drill.py --gateway-url http://127.0.0.1:$GPORT \\"
echo "        --model $MODEL --max-tokens 400 --kill-after 3.0"
echo
echo "  NOTE: the server and Ray head are LEFT RUNNING for the drill. Stop with:"
echo "    pkill -f 'vllm serve'; $VENV/bin/ray stop --force"
