#!/usr/bin/env bash
# Commit the doc pointers to the self-check scripts. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -F - <<'MSG'
docs: point the interconnect playbook at the readiness self-check

Section 1 now opens with the fast path: run scripts/check_node_readiness.py on each
machine in WSL and send back the printed "SEND THESE BACK TO THE CONTROLLER" block
(or the JSON). It collects every field the hand-filled table asked for plus the
heterogeneity verdicts, so nobody has to fill a table by hand.

The handover checklist in section 8 leads with the same command and keeps the manual
fallback for machines where Python or the repo is not set up yet.

The controller side is scripts/fleet_plan_from_readiness.py over the collected
node_readiness_*.json reports; it decides the fleet-wide stack version (lowest driver
CUDA), the model tier (smallest VRAM, with vLLM's even layer split), whether each
stage fits including the first/last stage's vocabulary cost, and which machines have
hard blockers.
MSG
git -c safe.directory='*' push origin main 2>&1 | tail -2
echo "pushed $(git rev-parse --short HEAD)"
echo
echo "=== final state ==="
git log --oneline -4
echo "tracked files: $(git ls-files | wc -l)"
git status -s; echo "(clean if nothing above)"
