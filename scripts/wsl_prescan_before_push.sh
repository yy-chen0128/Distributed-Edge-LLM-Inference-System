#!/usr/bin/env bash
# Before pushing: scan the UNPUSHED content for identifiers. Self-derived only --
# this script must not itself contain any literal identifier.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

ME="$(whoami)"
# derive the local address prefix (first two octets) instead of hardcoding it
PREFIX="$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9]+\.[0-9]+\.' | head -1)"
# derive the Windows profile name instead of hardcoding it
WINUSER=""
for d in /mnt/c/Users/*/; do
  b="$(basename "$d")"
  case "$b" in Public|Default|"Default User"|"All Users"|WDAGUtilityAccount) ;; *) WINUSER="$b"; break ;; esac
done

echo "=== derived (not hardcoded) identifiers to look for ==="
echo "  unix user      : $ME"
echo "  windows profile: ${WINUSER:-<none found>}"
echo "  local prefix   : ${PREFIX:-<unknown>}"

echo
echo "=== unpushed commits ==="
git log --oneline origin/main..main

echo
echo "=== files changed by the unpushed commits ==="
git diff --name-only origin/main..main

echo
echo "=== SCAN: identifiers inside the unpushed content ==="
hits=0
for f in $(git diff --name-only origin/main..main); do
  [ -f "$f" ] || continue
  for pat in "$ME" "$WINUSER" "$PREFIX"; do
    [ -z "$pat" ] && continue
    n=$(grep -c -i -F -- "$pat" "$f" 2>/dev/null || true)
    if [ "${n:-0}" -gt 0 ]; then
      echo "  HIT  $f : '$pat' x$n"
      grep -n -i -F -- "$pat" "$f" | head -4 | sed 's/^/        /'
      hits=$((hits + 1))
    fi
  done
done
echo "  total files with hits: $hits"

echo
echo "=== also scan the whole tracked tree (the leak we already knew about) ==="
for pat in "$ME" "$WINUSER"; do
  [ -z "$pat" ] && continue
  echo "  --- '$pat' in tracked files ---"
  git grep -l -i -F -- "$pat" 2>/dev/null | head -10
done
