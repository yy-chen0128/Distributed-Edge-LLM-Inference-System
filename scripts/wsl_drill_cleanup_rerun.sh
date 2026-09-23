#!/usr/bin/env bash
# Clean up the previous drill (it hung) and re-run with stall detection. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== kill leftovers from the hung drill ==="
pkill -f gateway_drill 2>/dev/null && echo "  killed drill" || echo "  no drill"
pkill -f vllm_gateway 2>/dev/null && echo "  killed gateway" || echo "  no gateway"
pkill -f 'vllm serve' 2>/dev/null && echo "  killed vllm" || echo "  no vllm"
pkill -f vllm_restart 2>/dev/null || true
sleep 6
pgrep -af 'gateway_drill|vllm_gateway|vllm serve' | head -3 || echo "  all clear"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  gpu used: /'

echo
echo "=== compile-check the gateway and clients ==="
"$HOME/venvs/pair/bin/python" -m py_compile edge_llm_scheduler/gateway/vllm_gateway.py \
   scripts/gateway_drill.py scripts/vllm_bench_prefix.py && echo "  OK" || exit 1

echo
echo "=== re-run the drill ==="
bash scripts/wsl_gateway_drill_all.sh 2>&1 | tail -45
