#!/usr/bin/env bash
# Commit the documentation-entry fix + check the install. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

git add -A
echo "=== staged ==="
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: make the per-machine setup doc unambiguous

There was a real maintenance gap: "how to set up a machine" and "how to run the
cluster" were spread over documents, and runbook-4-laptops.md (which is purely the
self-built-engine route: start_agent.ps1, cluster.json, run_real_pipeline) carried
no marker that it is no longer the main line -- and it was not even listed in the
docs index. A classmate following it would go down the wrong path.

Fixes:
- runbook-4-laptops.md gains a scope header stating it is the self-built-engine
  route (not the current main line) and pointing to environment-setup.md for
  per-machine setup and four-machine-interconnect.md for the current vLLM route.
- docs/README.md deploy section now opens with an entry table: "what I want to do
  -> which document to read -> which route", and states explicitly that
  environment-setup.md is the ONLY per-machine setup document (new content must be
  added there rather than in a new file, so readers cannot end up on a stale copy).
- the deploy table itself now lists every document, including runbook-4-laptops.md,
  with the main-line ones marked.
EOF
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -F "$MSGFILE" >/dev/null
rm -f "$MSGFILE"

echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -2

echo
echo "=== vLLM 0.10.1 install progress ==="
LOG=.models/logs/vllm10_install.log
if [ -f .models/logs/vllm10.pid ] && kill -0 "$(cat .models/logs/vllm10.pid)" 2>/dev/null; then
  echo "wrapper alive: pid $(cat .models/logs/vllm10.pid)"
fi
pgrep -af 'venvs/vllm/bin/pip' | head -2
echo "venv: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)"
echo "downloads so far: $(grep -c 'Downloading' "$LOG" 2>/dev/null)"
tail -4 "$LOG" 2>/dev/null
echo
df -h / /mnt/d | tail -2
