#!/usr/bin/env bash
# Re-run the four-phase cache test, now that the client's syntax error is fixed.
# Compile-checks the client FIRST so we do not burn another 10 minutes.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
PY="$HOME/venvs/vllm/bin/python"
OUT=.models/logs/v1_smoke

echo "=== compile check the client and the launcher ==="
"$PY" -m py_compile scripts/vllm_bench_prefix.py && echo "  vllm_bench_prefix.py: OK"
"$PY" -c "
import ast,sys
src=open('.models/logs/run_v1_smoke.sh').read()
print('  launcher is a shell script, skipping python check')
"
echo
echo "=== make sure no old server/smoke is still running ==="
pkill -f 'vllm serve' 2>/dev/null && echo "  killed leftover vllm serve" || echo "  none"
if [ -f .models/logs/v1_smoke.pid ]; then
  kill "$(cat .models/logs/v1_smoke.pid)" 2>/dev/null && echo "  killed old smoke driver" || true
fi
sleep 4

echo
echo "=== relaunch ==="
mv -f "$OUT/driver.log" "$OUT/driver.failed1.log" 2>/dev/null || true
nohup bash .models/logs/run_v1_smoke.sh > "$OUT/driver.log" 2>&1 &
echo $! > .models/logs/v1_smoke.pid
echo "started pid $(cat .models/logs/v1_smoke.pid)"
echo "poll: $OUT/driver.log"
