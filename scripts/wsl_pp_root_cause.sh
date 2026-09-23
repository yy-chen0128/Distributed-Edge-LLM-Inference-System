#!/usr/bin/env bash
# Extract the ROOT CAUSE of the PP failures (not just the tail). ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/v2_mp

for pp in 2 4; do
  log="$OUT/server_pp${pp}.log"
  echo "############ PP=$pp root cause ############"
  [ -f "$log" ] || { echo "  (no log)"; continue; }
  echo "--- lines around the FIRST error ---"
  grep -n -m1 -iE 'error|Traceback|RuntimeError|CUDA|NCCL|out of memory' "$log" | while IFS=: read -r ln rest; do
    sed -n "$((ln > 8 ? ln - 8 : 1)),$((ln + 22))p" "$log"
  done
  echo
  echo "--- all distinct error-ish lines (compressed) ---"
  grep -ioE '(CUDA error: [a-z ]+|out of memory[^"]*|NCCL[A-Za-z ]*|invalid device ordinal|No space left[a-z ]*|Address already in use|Connection refused|c10d[^ ]*)' "$log" \
    | sort | uniq -c | sort -rn | head -10
  echo
  echo "--- did any rank report starting? ---"
  grep -icE 'Starting vLLM|init engine|placement' "$log"
  echo
done

echo "=== GPU memory right now (did something OOM?) ==="
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo
echo "=== free RAM (4 ranks of KV/CPU buffers) ==="
free -g | sed 's/^/  /'
echo
echo "=== any leftover processes? ==="
pgrep -af 'vllm|ray::' | head -5 || echo "  none"
