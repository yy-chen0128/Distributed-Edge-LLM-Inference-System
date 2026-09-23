#!/usr/bin/env bash
# Where is the PP=4 (ray) startup? ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/v2b

echo "=== processes ==="
pgrep -af 'vllm serve' | head -3 || echo "  no vllm serve"
pgrep -af 'ray::' | head -3 || echo "  no ray workers"
"$HOME/venvs/vllm/bin/ray" status 2>/dev/null | sed -n '/Resources/,/Pending Demands/p' | head -8

echo
echo "=== how far did the server get? ==="
if [ -f "$OUT/server2.log" ]; then
  wc -l < "$OUT/server2.log" | sed 's/^/  log lines: /'
  grep -iE 'placement group|world_size|rank|init engine|Starting vLLM|Capturing|Available KV|error|Error|Traceback' \
      "$OUT/server2.log" | tail -14
  echo "  --- last 5 raw lines ---"
  tail -5 "$OUT/server2.log"
else
  echo "  (no server2.log)"
fi

echo
echo "=== can we already query it? ==="
curl -s -m 5 http://127.0.0.1:8020/v1/models | head -c 200 || echo "  (no response)"
echo
echo "=== GPU ==="
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
