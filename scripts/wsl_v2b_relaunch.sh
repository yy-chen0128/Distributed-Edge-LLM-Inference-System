#!/usr/bin/env bash
# Relaunch V2b (local Ray + PP=4) with the log dir created first. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
mkdir -p .models/logs/v2b
pkill -f 'vllm serve' 2>/dev/null || true
"$HOME/venvs/vllm/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 3

nohup bash .models/logs/run_v2b_ray.sh > .models/logs/v2b/driver.log 2>&1 &
echo $! > .models/logs/v2b.pid
echo "V2b started pid $(cat .models/logs/v2b.pid)"

echo
echo "=== waiting up to ~10 min for it to finish ==="
for i in $(seq 1 60); do
  if grep -q 'V2B_DONE' .models/logs/v2b/driver.log 2>/dev/null; then
    echo "finished after ~$((i * 10))s"; break
  fi
  if ! kill -0 "$(cat .models/logs/v2b.pid)" 2>/dev/null; then
    echo "process exited after ~$((i * 10))s"; break
  fi
  [ $((i % 6)) -eq 0 ] && printf '  [%3ds] %s\n' "$((i * 10))" \
      "$(tail -1 .models/logs/v2b/driver.log 2>/dev/null | cut -c1-80)"
  sleep 10
done

echo
echo "############ V2b RESULT ############"
cat .models/logs/v2b/driver.log 2>/dev/null | tail -45
echo
echo "=== ray status snapshot (if it started) ==="
tail -15 .models/logs/v2b/ray_start.log 2>/dev/null
