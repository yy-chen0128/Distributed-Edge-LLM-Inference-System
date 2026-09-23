#!/usr/bin/env bash
# V2 (revised): validate the PP>1 mechanism on ONE physical GPU using device
# aliasing, because Ray's oversubscription hands out ordinals that do not exist.
# CUDA_VISIBLE_DEVICES=0,0,0,0 exposes the same card four times, so torch sees 4
# devices and set_device(0..3) is legal. Performance is meaningless here -- this
# only proves the multi-rank PP path runs and produces correct output.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"; PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct
OUT=.models/logs/v2_mp; mkdir -p "$OUT"
export PYTHONPATH=.

echo "=== stop ray and any server ==="
pkill -f 'vllm serve' 2>/dev/null || true
"$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 3
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  gpu: /'

run_pp() {
  local PP="$1"
  local port="$2"
  local log="$OUT/server_pp${PP}.log"
  echo
  echo "########## PP=$PP (mp backend, aliased devices) ##########"
  CUDA_VISIBLE_DEVICES=0,0,0,0 nohup "$VENV/bin/vllm" serve "$MODEL" \
     --port "$port" --pipeline-parallel-size "$PP" \
     --distributed-executor-backend mp \
     --max-model-len 2048 --gpu-memory-utilization 0.30 \
     > "$log" 2>&1 &
  local pid=$!
  local ready=0
  for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    sleep 5
  done
  if [ "$ready" = "1" ]; then
    echo "  PP=$PP READY after ~$((i*5))s"
    grep -iE 'pipeline|PP|rank|world|init engine|Starting vLLM' "$log" | head -8
    "$PY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$port" \
        --prefix-tokens 300 --requests 2 --max-tokens 12 --label "PP${PP}-alias" 2>&1 | tail -5
  else
    echo "  PP=$PP NOT READY after $((i*5))s; last 12 lines:"
    tail -12 "$log"
  fi
  kill "$pid" 2>/dev/null; sleep 4
  pkill -f 'vllm serve' 2>/dev/null || true
  sleep 4
}

run_pp 2 8030
run_pp 4 8031

echo
echo "### V2MP_DONE"
