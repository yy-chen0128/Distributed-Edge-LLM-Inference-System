#!/usr/bin/env bash
# Start the AWQ (int4) 7B download in the background -- needed because 4 GB GPUs
# cannot hold 7 fp16 layers. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
mkdir -p .models/logs
PY="$HOME/venvs/pair/bin/python"

REPO=Qwen/Qwen2.5-7B-Instruct-AWQ
DEST=.models/Qwen2.5-7B-Instruct-AWQ

echo "=== disk before ==="
df -h /mnt/d | tail -1

if [ -d "$DEST" ] && [ -f "$DEST/config.json" ] && ! ls "$DEST"/*.part >/dev/null 2>&1; then
  echo "already present:"; du -sh "$DEST"; ls -1 "$DEST" | head -8
  exit 0
fi

if [ -f .models/logs/awq.pid ] && kill -0 "$(cat .models/logs/awq.pid)" 2>/dev/null; then
  echo "already downloading: pid $(cat .models/logs/awq.pid)"
else
  nohup "$PY" scripts/hf_mirror_download.py --repo "$REPO" --dest "$DEST" \
    > .models/logs/awq_download.log 2>&1 &
  echo $! > .models/logs/awq.pid
  echo "started: pid $(cat .models/logs/awq.pid)"
fi

sleep 25
echo
echo "--- log ---"
tail -c 300 .models/logs/awq_download.log 2>/dev/null
echo
echo "--- size so far ---"
du -sh "$DEST" 2>/dev/null || echo "(nothing yet)"
echo
df -h /mnt/d | tail -1
