#!/usr/bin/env bash
# What happened in the drill? ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/drill

echo "=== processes ==="
pgrep -af 'gateway_drill|vllm_gateway|vllm serve|vllm_restart' | head -6 || echo "  none"

echo
echo "=== gateway log: recovery decisions ==="
grep -aiE 'failed|replay|down|resume|forwarding|timeout|error' "$OUT/gateway.log" 2>/dev/null | tail -20
echo "  (total gateway log lines: $(wc -l < "$OUT/gateway.log" 2>/dev/null || echo 0))"

echo
echo "=== gateway status now ==="
curl -s -m 5 http://127.0.0.1:8100/admin/status || echo "  (gateway not responding)"

echo
echo
echo "=== server logs: was there a restart? ==="
for f in "$OUT/server.log" "$OUT/server_after_restart.log"; do
  [ -f "$f" ] || { echo "### $f: missing"; continue; }
  echo "### $(basename "$f")"
  echo "  lines: $(wc -l < "$f")"
  grep -aE 'Starting vLLM API server|init engine|Available KV|Engine core' "$f" | tail -4
done

echo
echo "=== restart helper output (if any) ==="
tail -3 .models/logs/vllm_restart.sh 2>/dev/null
ls -la "$OUT/" 2>/dev/null

echo
echo "=== is vLLM currently up? ==="
curl -s -m 5 http://127.0.0.1:8000/v1/models | head -c 120 || echo "  (not up)"
