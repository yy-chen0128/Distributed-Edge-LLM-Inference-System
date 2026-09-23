#!/usr/bin/env bash
# Poll the four-phase smoke test for up to ~5 minutes. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/v1_smoke

for i in $(seq 1 30); do
  if grep -q 'V1_SMOKE_DONE' "$OUT/driver.log" 2>/dev/null; then
    echo "=== SMOKE FINISHED ==="
    break
  fi
  if ! kill -0 "$(cat .models/logs/v1_smoke.pid 2>/dev/null)" 2>/dev/null; then
    echo "=== smoke process gone ==="
    break
  fi
  sleep 10
done

echo "--- phases seen ---"
grep -E '^#####' "$OUT/driver.log" 2>/dev/null
echo "--- ready / not-ready events ---"
grep -E 'ready|NOT READY|LMCache connector failed' "$OUT/driver.log" 2>/dev/null
echo
echo "--- bench results ---"
for f in "$OUT"/bench_*.log; do
  [ -f "$f" ] || continue
  echo "### $(basename "$f" .log)"
  grep -E '^\[|JSON' "$f" | tail -6
done
echo
echo "--- driver tail ---"
tail -12 "$OUT/driver.log" 2>/dev/null
echo
echo "--- lmcache activity in the C/D server logs (if present) ---"
for f in "$OUT"/server_C-lmcache.log "$OUT"/server_D-lmcache2.log; do
  [ -f "$f" ] || continue
  echo "### $(basename "$f")"
  grep -iE 'lmcache|kv_connector|cache' "$f" | tail -6
done
