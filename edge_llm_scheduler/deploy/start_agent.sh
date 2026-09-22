#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 一台笔记本上启动 stage agent（Linux / macOS）。
#
# 用法（在 project 目录下）：
#   bash edge_llm_scheduler/deploy/start_agent.sh alpha 9100 .models/Qwen2.5-0.5B-Instruct cuda:0
#
set -euo pipefail

NODE_ID="${1:-alpha}"
PORT="${2:-9100}"
MODEL="${3:-.models/Qwen2.5-0.5B-Instruct}"
DEVICE="${4:-cuda:0}"
DTYPE="${5:-auto}"
WIRE_DTYPE="${6:-float16}"
THREADS="${7:-4}"

cd "$(dirname "$0")/../.."
export PYTHONPATH="."
export PYTHONUNBUFFERED=1

echo "启动 agent node=${NODE_ID} port=${PORT} device=${DEVICE} model=${MODEL}"
exec python -m edge_llm_scheduler.agents.stage_agent \
  --node-id "${NODE_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --wire-dtype "${WIRE_DTYPE}" \
  --threads "${THREADS}"
