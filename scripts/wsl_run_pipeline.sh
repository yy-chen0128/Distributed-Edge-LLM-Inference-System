#!/usr/bin/env bash
# P1 milestone inside WSL: 4-stage pipeline on the GPU.
# ASCII ONLY.
set -u
PY="$HOME/venvs/pair/bin/python"
PROJ="$(find /mnt/d/Newproject -maxdepth 2 -type d -name project 2>/dev/null | head -1)"
[ -z "$PROJ" ] && { echo "PROJECT_NOT_FOUND"; exit 1; }
cd "$PROJ" || exit 1
export PYTHONPATH=.
MODEL=".models/Qwen2.5-0.5B-Instruct"

echo "[model] $MODEL"
ls "$MODEL"/config.json >/dev/null 2>&1 || { echo "MODEL_MISSING"; exit 1; }

echo "[run] 4 local agents on cuda:0"
"$PY" -m edge_llm_scheduler.experiments.run_real_pipeline \
  --local 4 --device cuda:0 --dtype auto --wire-dtype float16 \
  --model "$MODEL" --max-tokens 6 \
  --metrics-out .models/wsl_gpu_pipeline.json 2>&1 | tail -28
