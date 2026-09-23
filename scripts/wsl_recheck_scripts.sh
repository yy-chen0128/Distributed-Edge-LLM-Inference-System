#!/usr/bin/env bash
# Re-check syntax of all two-node scripts, amend the unpushed commit. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== syntax check every shell script in scripts/ (catches this class of bug) ==="
bad=0
for f in scripts/*.sh; do
  if ! bash -n "$f" 2>/tmp/n.err; then
    echo "  FAIL $(basename "$f"): $(head -1 /tmp/n.err)"
    bad=$((bad + 1))
  fi
done
echo "  checked $(ls scripts/*.sh | wc -l) scripts, failures: $bad"
[ "$bad" -gt 0 ] && exit 1

echo
echo "=== python compile check ==="
"$HOME/venvs/pair/bin/python" -m py_compile scripts/*.py edge_llm_scheduler/gateway/*.py && echo "  OK"

echo
echo "=== amend the unpushed commit ==="
git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q --amend --no-edit
git log --oneline -2
echo
echo "=== unsaved-from-remote (still deliberately held) ==="
git log --oneline origin/main..main
