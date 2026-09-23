#!/usr/bin/env bash
# Commit the verification helper locally (still NOT pushing: privacy work is paused).
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -m "chore: script that verifies the peer-agent brief is untracked and sanitized

The handoff brief for the agent on the other machine lives OUTSIDE the repository
(workspace root, next to but not inside project/) so git can never track it, and it
uses placeholders for every machine-specific value. This script asserts both: that
the path is outside the repo, that git ls-files cannot see it, and that a scan for
the local username, the Windows profile name, and the campus address ranges finds
nothing. It exists so the check is repeatable rather than a one-off claim."
git log --oneline -2
echo
echo "held (not pushed): $(git log --oneline origin/main..main | wc -l) commit(s)"
