#!/usr/bin/env bash
# Commit and push the policy + batching docs. ASCII ONLY (PowerShell pipe re-encodes).
set -eu
cd "$HOME/pair"

echo "=== git status before ==="
git status --short

git add -A

echo
echo "=== staged ==="
git diff --cached --stat

MSG=$(cat <<'EOF'
docs: policy principles/testability + single-machine batching scope

- add docs/research/policies-principles-and-testability.md
  five policies (E2Placement, CapabilityPlacement, PriorityMigration,
  CapabilityReparallelization, TokenRecovery): principle, exact algorithm,
  parameters, code locations, measured evidence, and what is/is not testable
- add docs/design/single-machine-scope-and-batching.md
  batching vs micro-batching vs DP; measured proof that per-call overhead
  dominates model compute; engine/protocol/controller changes; acceptance
- add scripts/probe_policies.py (behaviour probe, evidence for the above)
- add scripts/wsl_run_policy_sim.sh
- fix: flag that the K=1..4 throughput row is a CPU fp32 run, and that the
  GPU run had all four agents on cuda:0 (shared GPU)
- docs: drop audience-directed phrasing (no "for students" framing)
EOF
)

git commit -m "$MSG"

echo
echo "=== commit ==="
git log --oneline -1
git show --stat --oneline HEAD | tail -20

echo
echo "=== push ==="
git push origin main
git log --oneline -3
