#!/usr/bin/env bash
# Run on the WORKER machine (B): join the Ray cluster of the head machine.
#
#   HEAD_IP=<A LAN IP> bash scripts/two_node_worker.sh
#
# It auto-detects the interface used to reach the head and exports NCCL/GLOO
# socket names + VLLM_HOST_IP from it. This matters: WSL exposes several
# interfaces and NCCL otherwise picks the wrong one; and the vLLM Ray executor
# builds a node-affinity label from VLLM_HOST_IP, so a wrong value (e.g.
# 127.0.0.1) makes the placement group unsatisfiable.
#
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="${VENV:-$HOME/venvs/vllm}"
# NOTE: no apostrophes inside ${VAR:?...} -- bash mis-parses them and the whole
# script fails with "unexpected EOF while looking for matching quote".
HEAD_IP="${HEAD_IP:?set HEAD_IP to the LAN IP of the head machine}"
RAY_PORT="${RAY_PORT:-6379}"
MODEL="${MODEL:-.models/Qwen2.5-0.5B-Instruct}"

echo "=== this machine ==="
echo "  hostname : $(hostname)"
echo "  addrs    : $(ip -4 addr show 2>/dev/null | grep -oP 'inet \K[0-9.]+' | tr '\n' ' ')"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  gpu      : /'

echo
echo "=== route to the head -> which interface / source IP? ==="
ROUTE="$(ip route get "$HEAD_IP" 2>/dev/null | head -1)"
echo "  $ROUTE"
IFACE="$(echo "$ROUTE" | grep -oP 'dev \K[^ ]+' | head -1)"
SRCIP="$(echo "$ROUTE" | grep -oP 'src \K[0-9.]+' | head -1)"
[ -z "$IFACE" ] && { echo "  cannot route to $HEAD_IP -> fix networking first"; exit 1; }
export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"
export VLLM_HOST_IP="$SRCIP"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME  GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME  VLLM_HOST_IP=$VLLM_HOST_IP"

echo
echo "=== reachability ==="
if ping -c 3 -W 2 "$HEAD_IP" >/dev/null 2>&1; then echo "  ping $HEAD_IP: ok"; else echo "  ping $HEAD_IP: FAILED (campus WiFi client isolation?)"; fi
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$HEAD_IP/$RAY_PORT" 2>/dev/null; then
  echo "  TCP $HEAD_IP:$RAY_PORT: open (the head is running)"
else
  echo "  TCP $HEAD_IP:$RAY_PORT: closed -> start the head first (two_node_head.sh)"
fi

echo
echo "=== model present locally? (each PP rank loads its own shard from ITS disk) ==="
if [ -d "$MODEL" ]; then
  echo "  ok: $MODEL  ($(du -sh "$MODEL" 2>/dev/null | cut -f1))"
else
  echo "  MISSING: $MODEL -> download it here first:"
  echo "    python scripts/hf_mirror_download.py --repo Qwen/Qwen2.5-0.5B-Instruct --dest $MODEL"
  exit 1
fi

echo
echo "=== join the Ray cluster ==="
"$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 3
"$VENV/bin/ray" start --address="$HEAD_IP:$RAY_PORT" --num-gpus=1 \
   --min-worker-port=10002 --max-worker-port=10100 2>&1 | tail -8
sleep 5

echo
echo "=== cluster view from here ==="
"$VENV/bin/ray" status 2>&1 | sed -n '1,20p'

echo
echo "=== keep this shell alive so Ray stays up; press Ctrl-C to leave ==="
echo "  (the worker must also be able to receive on ports 10002-10100)"
