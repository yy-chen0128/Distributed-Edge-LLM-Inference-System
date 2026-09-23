#!/usr/bin/env bash
# Wait for the four-phase test to finish and print the numbers. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/v1_smoke

echo "=== waiting (up to ~13 min) ==="
for i in $(seq 1 78); do
  if grep -q 'V1_SMOKE_DONE' "$OUT/driver.log" 2>/dev/null; then
    echo "finished after ~$((i * 10))s"; break
  fi
  if ! kill -0 "$(cat .models/logs/v1_smoke.pid 2>/dev/null)" 2>/dev/null; then
    echo "smoke process exited"; break
  fi
  if [ $((i % 6)) -eq 0 ]; then
    printf '  [%3ds] phases: %s | last: %s\n' "$((i * 10))" \
      "$(grep -c '^#####' "$OUT/driver.log" 2>/dev/null)" \
      "$(tail -1 "$OUT/driver.log" 2>/dev/null | cut -c1-70)"
  fi
  sleep 10
done

echo
echo "############ FINAL RESULTS ############"
for f in "$OUT"/bench_*.log; do
  [ -f "$f" ] || continue
  echo "### $(basename "$f" .log)"
  grep -E '^\[' "$f" | tail -7
  echo
done

echo "=== phases and readiness ==="
grep -E '^#####|ready|NOT READY|LMCache connector failed' "$OUT/driver.log" 2>/dev/null

echo
echo "=== LMCache cache-hit evidence (phases C/D) ==="
for f in "$OUT"/server_C-lmcache.log "$OUT"/server_D-lmcache2.log; do
  [ -f "$f" ] || continue
  echo "### $(basename "$f")"
  grep -ioE '(hit|miss|retriev|stored|lookup)[a-z_ ]{0,40}' "$f" | sort | uniq -c | sort -rn | head -8
done

echo
echo "=== driver tail ==="
tail -15 "$OUT/driver.log" 2>/dev/null
