#!/usr/bin/env bash
# READ ONLY. Focused: does the ALREADY PUBLIC origin/main contain the identifiers,
# and do the two UNPUSHED commits newly introduce any? ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

ME="$(whoami)"
PREFIX="$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9]+\.[0-9]+\.' | head -1)"
WINUSER=""
for d in /mnt/c/Users/*/; do
  b="$(basename "$d")"
  case "$b" in Public|Default|"Default User"|"All Users"|WDAGUtilityAccount) ;; *) WINUSER="$b"; break ;; esac
done

label() {
  if [ "$1" = "$ME" ]; then echo "unix-user"
  elif [ "$1" = "$WINUSER" ]; then echo "win-profile"
  else echo "net-prefix"; fi
}

echo "=== A. identifiers already inside the PUBLIC origin/main tree ==="
for pat in "$ME" "$WINUSER" "$PREFIX"; do
  [ -z "$pat" ] && continue
  files="$(git grep -l -i -F -- "$pat" origin/main -- . 2>/dev/null)"
  n=$(printf '%s' "$files" | grep -c . || true)
  echo "  [$(label "$pat")] origin/main tracked files: ${n:-0}"
  [ "${n:-0}" -gt 0 ] && printf '%s\n' "$files" | head -12 | sed 's/^/        /'
done

echo
echo "=== B. the two unpushed commits: which files did they touch ==="
git log --oneline --name-only origin/main..main | sed 's/^/  /'

echo
echo "=== C. do the unpushed commits ADD identifier lines (tip tree is already clean) ==="
tmp=$(mktemp)
git diff origin/main..main > "$tmp" 2>/dev/null
for pat in "$ME" "$WINUSER" "$PREFIX"; do
  [ -z "$pat" ] && continue
  add=$(grep -i -F -- "$pat" "$tmp" | grep -c '^+' || true)
  del=$(grep -i -F -- "$pat" "$tmp" | grep -c '^-' || true)
  echo "  [$(label "$pat")] added lines: ${add:-0}   removed lines: ${del:-0}"
  grep -i -F -- "$pat" "$tmp" | grep '^+' | head -6 | cut -c1-120 | sed 's/^/        /'
done
rm -f "$tmp"

echo
echo "=== D. which files in the public tree carried the identifiers (per-pattern) ==="
for pat in "$ME" "$WINUSER"; do
  [ -z "$pat" ] && continue
  echo "  [$(label "$pat")]"
  git grep -c -i -F -- "$pat" origin/main -- . 2>/dev/null | head -12 | sed 's/^/        /'
done

echo
echo "=== E. history commits that mention them (main only, cheap) ==="
git log --oneline main | head -40 | sed 's/^/  /'
echo "DONE"
