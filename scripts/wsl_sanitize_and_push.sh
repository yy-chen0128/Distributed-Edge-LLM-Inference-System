#!/usr/bin/env bash
# Gate, then publish. Run this instead of a bare "git push".
#
# Why it exists:
#   * Two commits made earlier in this session (the two-machine bring-up and the
#     brief-verify script) were NOT yet pushed, and they introduced the literal
#     identifiers the privacy scanners were supposed to find. Scanning them would
#     have leaked the identifiers again. They are local-only, so instead of adding a
#     "sanitize" commit on top (which would publish the literals forever) this squash
#     rewrites them into one clean commit. No force-push is needed: the remote never
#     saw them, so the result is still a fast-forward.
#   * Everything is derived at runtime; this file contains no literal identifier.
#
# Usage:  bash scripts/wsl_sanitize_and_push.sh          # verify only, no writes
#         bash scripts/wsl_sanitize_and_push.sh --push    # actually push
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
PUSH=no
[ "${1:-}" = "--push" ] && PUSH=yes

ME="$(whoami)"
PREFIX="$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9]+\.[0-9]+\.' | head -1)"
WINUSER=""
for d in /mnt/c/Users/*/; do
  b="$(basename "$d")"
  case "$b" in
    Public|Default|"Default User"|"All Users"|WDAGUtilityAccount) ;;
    *) WINUSER="$b"; break ;;
  esac
done

echo "=== 0. syntax-check every shell script about to be published ==="
bad=0
for f in scripts/*.sh; do
  if ! bash -n "$f" 2>/dev/null; then echo "  SYNTAX FAIL: $f"; bad=$((bad + 1)); fi
done
echo "  scripts with syntax errors: $bad"
[ "$bad" -gt 0 ] && { echo "  refusing to push"; exit 1; }

echo
echo "=== 1. refresh the remote-tracking ref (never trust a stale origin/main) ==="
git -c safe.directory='*' fetch -q origin main || { echo "  fetch failed"; exit 1; }
echo "  origin/main : $(git rev-parse origin/main)"
echo "  local main  : $(git rev-parse main)"
unpushed=$(git rev-list --count origin/main..main)
echo "  unpushed commits: $unpushed"

echo
echo "=== 2. stage everything and scan ONLY what the push would ADD ==="
git add -A
hits=0
for pat in "$ME" "$WINUSER" "$PREFIX"; do
  [ -z "$pat" ] && continue
  n=$(git diff --cached origin/main 2>/dev/null | grep '^+' | grep -c -i -F -- "$pat" || true)
  echo "  added lines matching one derived identifier: ${n:-0}"
  if [ "${n:-0}" -gt 0 ]; then
    git diff --cached origin/main | grep '^+' | grep -i -F -- "$pat" | head -5 | cut -c1-110 | sed 's/^/      /'
    hits=$((hits + 1))
  fi
done
if [ "$hits" -gt 0 ]; then
  echo "  !! refusing to publish: sanitize the staged content first"
  exit 1
fi
echo "  clean."

echo
echo "=== 3. what would be published ==="
git diff --cached --stat origin/main | tail -20
echo "  files: $(git diff --cached --name-only origin/main | wc -l)"

if [ "$PUSH" != "yes" ]; then
  echo
  echo "  DRY RUN (no commit, no push). Re-run with --push to publish."
  echo "  NOTE: the index is now staged; 'git reset' if you want it back."
  exit 0
fi

echo
echo "=== 4. squash the unpushed commits into one clean commit ==="
if [ "$unpushed" -gt 1 ]; then
  git reset --soft origin/main || exit 1
  echo "  soft-reset to origin/main; all content preserved in the index"
fi
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -F - <<'MSG'
feat: two-machine bring-up, prefix-scaling bench client, self-derived privacy gate

Prepares the second machine for the fleet and adds the checks that must pass before
anything is published.

Two-machine bring-up (scripts/two_node_head.sh, two_node_worker.sh,
two_node_this_machine_info.sh + docs/deploy/two-machine-bringup.md)
- head: PP=1 baseline, then PP=2 over Ray, and prints the same-machine vs
  cross-machine per-token delta so the interconnect cost is visible, not assumed
- worker: verifies ping, the head's Ray port and the local model, then joins with
  ray start --address
- both derive the NCCL/GLOO interface and the source address for VLLM_HOST_IP at
  runtime; a hardcoded 127.0.0.1 there makes the placement group unsatisfiable
- the info script prints this machine's bridge public key, so the peer can authorize
  it without copying a private file around
- runbook carries the model-fit table (0.5B and 3B fit, 7B fp16 does not at 7.2 GB
  usable, 7B AWQ int4 does) and the four prerequisites

Bench client (scripts/vllm_bench_prefix.py)
- reports prompt/completion token counts and TPOT, prints the server's error body
  instead of a bare HTTP 400, and supports --ignore-eos for fixed-length runs
- the prefix is built from " the": the previous filler word tokenised to several
  tokens each, so a nominal 3000-token prompt blew past max-model-len 8192

Verification (scripts/wsl_verify_peer_brief.sh, wsl_commit_brief_verify.sh)
- asserts the peer-agent brief stays outside the repo (git can never track it) and
  contains no real identifiers

Privacy gate (scripts/wsl_privacy_scan.sh, wsl_prescan_before_push.sh,
wsl_diag_before_push.sh, wsl_sanitize_and_push.sh)
- every identifier is derived at runtime from the machine: the unix user, the first
  two octets of the default-route address, and the non-system entry under
  /mnt/c/Users. The earlier versions hardcoded the strings they searched for, which
  turned the scanner itself into the leak; that is fixed here and the check is now
  documented in the file header
- wsl_sanitize_and_push.sh is the gate: syntax-check, fetch, stage, and refuse to
  publish if any added line matches a derived identifier. Default is a dry run

Also simplifies the history scan to main (--all dragged in a stale remote and took
minutes), and fixes the shell-quoting defects the two-node scripts had on first run
(apostrophes inside ${VAR:?} messages, nested single quotes in a grep).

Note: the git history STILL contains identifiers from commits that were pushed
before this session. Removing those needs a rewrite plus a force-push and is
deliberately not part of this change.
MSG
echo "  committed: $(git rev-parse --short HEAD)"

echo
echo "=== 5. push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -3
echo "  local  main: $(git rev-parse main)"
echo "  remote main: $(git -c safe.directory='*' ls-remote origin main | cut -f1)"

echo
echo "=== 6. post-push scan of the published tree ==="
for pat in "$ME" "$WINUSER"; do
  [ -z "$pat" ] && continue
  echo "  tracked files still matching a derived identifier: $(git grep -l -i -F -- "$pat" -- . 2>/dev/null | wc -l)"
done
echo "  (any remaining hits live in history, not in the tree)"
echo "DONE"
