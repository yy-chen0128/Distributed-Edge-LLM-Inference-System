#!/usr/bin/env bash
# Quick progress probe: is pip actually downloading (cache growth)? ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== pip cache size (wheels land here first) ==="
du -sh /mnt/d/pipcache 2>/dev/null
S1=$(du -sm /mnt/d/pipcache 2>/dev/null | cut -f1)
sleep 15
S2=$(du -sm /mnt/d/pipcache 2>/dev/null | cut -f1)
echo "growth over 15s: ${S1}MB -> ${S2}MB"

echo
echo "=== newest cached file ==="
find /mnt/d/pipcache -type f -newermt '-2 minutes' 2>/dev/null | tail -3

echo
echo "=== log: what is being downloaded right now ==="
tail -c 300 .models/logs/vllm_install2.log

echo
echo "=== venv / disk ==="
du -sh "$HOME/venvs/vllm" 2>/dev/null
df -h / /mnt/d | tail -2
