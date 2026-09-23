#!/usr/bin/env bash
# Wait for the local_disk test and print the decisive comparison. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/v1_disk

echo "=== waiting for V1_DISK_DONE (up to ~9 min) ==="
for i in $(seq 1 54); do
  if grep -q 'V1_DISK_DONE' "$OUT/driver.log" 2>/dev/null; then
    echo "finished after ~$((i * 10))s"; break
  fi
  if ! kill -0 "$(cat .models/logs/v1_disk.pid 2>/dev/null)" 2>/dev/null; then
    echo "process exited"; break
  fi
  [ $((i % 6)) -eq 0 ] && printf '  [%3ds] %s\n' "$((i * 10))" \
      "$(tail -1 "$OUT/driver.log" 2>/dev/null | cut -c1-70)"
  sleep 10
done

echo
echo "############ RESULT ############"
for f in "$OUT"/bench_*.log; do
  [ -f "$f" ] || continue
  echo "### $(basename "$f" .log)"
  grep -E '^\[' "$f" | tail -6
  echo
done

echo "=== disk cache dir ==="
du -sh /mnt/d/lmcache_disk 2>/dev/null
find /mnt/d/lmcache_disk -maxdepth 2 2>/dev/null | head -8

echo
echo "=== LMCache hit/miss evidence in D2 (the after-restart phase) ==="
for f in "$OUT"/server_D2.log; do
  [ -f "$f" ] || continue
  grep -iE 'hit|miss|retriev|stored|lookup|disk' "$f" | tail -10
done

echo
echo "=== driver tail ==="
tail -18 "$OUT/driver.log" 2>/dev/null
