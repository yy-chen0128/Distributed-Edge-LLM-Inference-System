#!/usr/bin/env bash
# V2: prove that vLLM's pipeline parallelism works, on ONE physical GPU.
#
# Trick: CUDA_VISIBLE_DEVICES=0,0,0,0 exposes the same physical GPU four times,
# so vLLM sees 4 devices and can run --pipeline-parallel-size 4. Each rank then
# aliases physical GPU 0, so the memory footprint is ~4x and the performance
# numbers are MEANINGLESS -- this only validates the mechanism (ranks come up,
# activations flow between ranks, output is correct).
#
# ASCII ONLY.
set -u
VENV="${VENV:-$HOME/venvs/vllm}"
MODEL="${MODEL:-.models/Qwen2.5-0.5B-Instruct}"
PP="${PP:-4}"
PORT="${PORT:-8010}"
LOGDIR=".models/logs"
mkdir -p "$LOGDIR"

cd "$HOME/pair" || exit 1
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "NO_VLLM_VENV"; exit 1; }

echo "=== single-GPU PP=$PP (aliased devices; mechanism check only) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

LOG="$LOGDIR/vllm_pp${PP}.log"
CUDA_VISIBLE_DEVICES=0,0,0,0 nohup "$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --pipeline-parallel-size "$PP" \
  --distributed-executor-backend mp \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.35 \
  > "$LOG" 2>&1 &
PID=$!

ready=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done

if [ "$ready" = "1" ]; then
  echo "--- PP=$PP server ready ---"
  "$PY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$PORT" \
    --prefix-tokens 512 --requests 2 --label "PP${PP}"
  echo "--- rank placement seen by the server: ---"
  grep -iE "pipeline|rank|PP|worker" "$LOG" | head -15
else
  echo "--- PP=$PP did NOT come up; last log lines ---"
  tail -n 30 "$LOG"
  echo "--- fallback idea: --distributed-executor-backend ray with a Ray that"
  echo "    reports 4 GPUs (ray start --head --num-gpus=4) ---"
fi

kill "$PID" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
echo "=== done (remember: performance numbers here are not meaningful) ==="
