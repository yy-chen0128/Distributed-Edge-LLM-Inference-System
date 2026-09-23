#!/usr/bin/env bash
# Verify the peer-agent brief: (1) outside the repo / untracked, (2) no real
# identifiers in it.
#
# IMPORTANT: the identifiers it searches for are DERIVED at runtime. Never hardcode
# them here -- a scanner full of literals is itself the leak (that mistake was made
# once and caught before pushing).
#
# ASCII ONLY.
set -u
PROJ="$(find /mnt/d/Newproject -maxdepth 2 -type d -name project 2>/dev/null | head -1)"
WS="$(dirname "$PROJ")"
BRIEF="$WS/peer-agent-brief.md"

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

echo "=== where it lives ==="
echo "  workspace : $WS"
echo "  repo      : $PROJ"
echo "  brief     : $BRIEF"
[ -f "$BRIEF" ] || { echo "  MISSING"; exit 1; }
echo "  size      : $(wc -c < "$BRIEF") bytes, $(wc -l < "$BRIEF") lines"

echo
echo "=== is it inside the repo? (must NOT be) ==="
case "$BRIEF" in
  "$PROJ"/*) echo "  INSIDE the repo -> move it out, or it risks being tracked" ;;
  *) echo "  outside the repo -> git can never track it" ;;
esac
echo "  tracked by git? $(git -C "$PROJ" ls-files --error-unmatch "$BRIEF" >/dev/null 2>&1 && echo YES-BAD || echo no)"

echo
echo "=== identifier scan (all derived, expect zero hits) ==="
for pat in "$ME" "$WINUSER" "$PREFIX"; do
  [ -z "$pat" ] && continue
  n=$(grep -c -i -F -- "$pat" "$BRIEF" 2>/dev/null || true)
  printf '  %-18s hits=%s\n' "$pat" "${n:-0}"
done
echo "  (labels above are the derived values, printed only to this terminal)"

echo
echo "=== placeholders it does contain ==="
grep -o '<[A-Za-z-]*>' "$BRIEF" 2>/dev/null | sort | uniq -c | sort -rn | head -10
