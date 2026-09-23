#!/usr/bin/env bash
# Verify the git push state for real (no guessing). ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== local vs remote-tracking ==="
git log --oneline -6
echo
echo "origin/main = $(git rev-parse --short origin/main 2>/dev/null)"
echo "local  main = $(git rev-parse --short main 2>/dev/null)"
echo
echo "=== unpushed commits (empty = all pushed) ==="
git log --oneline origin/main..main 2>/dev/null || true
echo "(nothing above = nothing pending)"
echo
echo "=== working tree ==="
git status -s
echo "(nothing above = clean)"
echo
echo "=== remote branches/refs on the actual remote (not the cached ref) ==="
git -c safe.directory='*' ls-remote origin main 2>&1 | head -3
echo
echo "=== last commit detail ==="
git log -1 --format='%h %cd %s' --date=iso
echo
echo "=== how many commits total / tracked files ==="
git rev-list --count main
git ls-files | wc -l
