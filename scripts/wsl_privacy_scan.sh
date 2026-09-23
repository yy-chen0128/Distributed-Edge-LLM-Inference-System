#!/usr/bin/env bash
# Privacy scan: find personal identifiers in the repo (tracked files AND history).
#
# IMPORTANT: this script must never contain a literal identifier itself -- that is
# exactly how a "privacy scanner" becomes a leak. Everything it looks for is derived
# at runtime from the current machine.
#
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

ME="$(whoami)"
# first two octets of the address used for the default route (no hardcoded range)
PREFIX="$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9]+\.[0-9]+\.' | head -1)"
# the Windows profile directory name (no hardcoded name)
WINUSER=""
for d in /mnt/c/Users/*/; do
  b="$(basename "$d")"
  case "$b" in
    Public|Default|"Default User"|"All Users"|WDAGUtilityAccount) ;;
    *) WINUSER="$b"; break ;;
  esac
done

echo "=== derived identifiers (never literal in this file) ==="
echo "  unix user      : $ME"
echo "  windows profile: ${WINUSER:-<none>}"
echo "  local prefix   : ${PREFIX:-<unknown>}"

echo
echo "############ 1. TRACKED FILES containing them ############"
for pat in "$ME" "$WINUSER"; do
  [ -z "$pat" ] && continue
  echo "--- '$pat' ---"
  git grep -l -i -F -- "$pat" -- . 2>/dev/null | head -30
  echo "  files: $(git grep -l -i -F -- "$pat" -- . 2>/dev/null | wc -l)"
done

echo
echo "############ 2. address prefixes in tracked files ############"
if [ -n "$PREFIX" ]; then
  git grep -n -F -- "$PREFIX" -- . 2>/dev/null | head -20
  echo "  files: $(git grep -l -F -- "$PREFIX" -- . 2>/dev/null | wc -l)"
fi
echo "--- ssh key filenames / host paths (informational) ---"
git grep -l -E 'agent_ed25519|known_hosts|/mnt/c/Users/' -- . 2>/dev/null | head -20

echo
echo "############ 3. GIT HISTORY: commits that ever contained them ############"
for pat in "$ME" "$WINUSER"; do
  [ -z "$pat" ] && continue
  echo "--- '$pat' ---"
  # main only: --all drags in the stale 'legacy' remote and takes minutes
  git log --oneline -S"$pat" main | head -20
  echo "  commits on main: $(git log --oneline -S"$pat" main | wc -l)"
done

echo
echo "############ 4. what the remote has right now ############"
git -c safe.directory='*' ls-remote origin main
echo "  (identifiers in history require a rewrite + force-push to remove)"
