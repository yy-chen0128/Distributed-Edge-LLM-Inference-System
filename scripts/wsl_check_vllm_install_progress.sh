#!/usr/bin/env bash
# Is the vLLM install actually progressing? ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== pip processes ==="
ps -eo pid,etime,rss,cmd | grep -E 'pip|vllm' | grep -v grep | head -5

echo
echo "=== pip cache growth (wheels land here first) ==="
du -sh /mnt/d/pipcache 2>/dev/null || echo "(no cache dir yet)"
ls -1 /mnt/d/pipcache 2>/dev/null | wc -l

echo
echo "=== newest files in the cache ==="
find /mnt/d/pipcache -type f -newermt '-3 minutes' 2>/dev/null | head -5
echo "(files modified in the last 3 minutes above = download is moving)"

echo
echo "=== venv size ==="
du -sh "$HOME/venvs/vllm" 2>/dev/null

echo
echo "=== log tail (script buffers pip output through tail, so it may be empty) ==="
tail -c 300 .models/logs/vllm_install.log
echo
echo "=== pip download activity: count cache files twice, 10s apart ==="
A=$(find /mnt/d/pipcache -type f 2>/dev/null | wc -l)
S1=$(du -sm /mnt/d/pipcache 2>/dev/null | cut -f1)
sleep 10
B=$(find /mnt/d/pipcache -type f 2>/dev/null | wc -l)
S2=$(du -sm /mnt/d/pipcache 2>/dev/null | cut -f1)
echo "files: $A -> $B ; size MB: $S1 -> $S2"
