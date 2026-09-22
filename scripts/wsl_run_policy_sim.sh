#!/usr/bin/env bash
# Run policy-level simulation (mock, no GPU) to see what is testable today.
# ASCII ONLY.
set -u
PY="$HOME/venvs/pair/bin/python"
PROJ="$(find /mnt/d/Newproject -maxdepth 2 -type d -name project 2>/dev/null | head -1)"
[ -z "$PROJ" ] && { echo "PROJECT_NOT_FOUND"; exit 1; }
cd "$PROJ" || exit 1
export PYTHONPATH=.

echo "=== [1] placement policy comparison (mock, 3 nodes) ==="
"$PY" -m edge_llm_scheduler.experiments.run_simulation \
  --policy default e2 capability --nodes 3 2>&1 | tail -12

echo
echo "=== [2] layered pipeline control-flow simulation ==="
"$PY" -m edge_llm_scheduler.experiments.run_layered_simulation --nodes 3 2>&1 | tail -14

echo
echo "=== [3] policy-related unit tests ==="
"$PY" -m pytest edge_llm_scheduler/tests/test_e2_placement.py \
  edge_llm_scheduler/tests/test_capability_placement.py \
  edge_llm_scheduler/tests/test_priority_migration.py \
  edge_llm_scheduler/tests/test_reparallelization_recovery.py \
  edge_llm_scheduler/tests/test_pipeline_reconfiguration.py -q -p no:cacheprovider 2>&1 | tail -4
