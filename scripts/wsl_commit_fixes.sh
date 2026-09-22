#!/usr/bin/env bash
# Commit the policy fixes + new docs. ASCII ONLY (PowerShell pipe re-encodes).
set -eu
cd "$HOME/pair"

echo "=== status ==="
git status --short

git add -A

echo
echo "=== staged ==="
git diff --cached --stat

MSG=$(cat <<'EOF'
fix: 8 policy/scheduler defects; add real-workload dataset + batching measurement

Fixes (each has a regression test in tests/test_policy_fixes.py):
- E2Placement: UnboundLocalError when prompt is None but hit_tokens > 0
- PriorityMigration: spread blocks across targets (was: all to one node,
  because state.load is static during decide())
- MockKVStore.estimate_move_cost: was ignoring src/dst entirely; now uses the
  slower of the two links plus an RTT (local 0.14ms / remote 5.0ms)
- PriorityMigration: reuse_count == 0 made priority 0 regardless of prefill
  cost; added min_reuse_count (default 1)
- HFLayeredEngine: resume with no local KV cache used to feed the last token
  against an empty cache (wrong output); now recomputes the full prompt
- TaskScheduler: recovered result was discarded on NODE_LEFT; now stored and
  a TASK_DONE event is published
- TaskScheduler: hit_tokens was overwritten on every stage; now first stage only
- CapabilityReparallelization: comment said "remainder to largest weight",
  code uses largest fractional remainder (comment fixed)

New:
- docs/research/real-workload-datasets.md (real traces/datasets for arrival,
  prefix sharing, length distribution, churn; with gates, licenses, pitfalls
  and citable prefix-sharing numbers)
- docs/design/single-machine-scope-and-batching.md rewritten: exact test
  configuration (every real run was batch=1), measured c0/c1 split with CUDA
  synchronisation, and a correction that the 4-stage prefill numbers were
  cold-start, not steady state
- scripts/measure_stage_scaling.py (prefill curve, decode steady state,
  call-granularity upper bound)
- scripts/probe_policies.py, scripts/wsl_run_policy_sim.sh

Docs: drop audience-directed phrasing.
EOF
)

git commit -m "$MSG"

echo
echo "=== commit ==="
git log --oneline -3
