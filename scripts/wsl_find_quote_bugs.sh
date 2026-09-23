#!/usr/bin/env bash
# Find lines whose single-quote count is odd (the likely syntax culprits).
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

for f in scripts/two_node_worker.sh scripts/two_node_head.sh; do
  echo "=== $f ==="
  n=0
  while IFS= read -r line; do
    n=$((n + 1))
    # count single quotes in the line
    cnt=$(printf '%s' "$line" | tr -cd "'" | wc -c)
    if [ $((cnt % 2)) -ne 0 ]; then
      printf '  line %3d (quotes=%d): %s\n' "$n" "$cnt" "$line"
    fi
  done < "$f"
  echo
done
