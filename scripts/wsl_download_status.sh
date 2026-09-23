#!/usr/bin/env bash
# Report download progress for both jobs. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== [1] AWQ model ==="
if [ -f .models/logs/awq.pid ] && kill -0 "$(cat .models/logs/awq.pid)" 2>/dev/null; then
  echo "pid $(cat .models/logs/awq.pid): ALIVE"
else
  echo "pid file: NOT RUNNING"
fi
du -sh .models/Qwen2.5-7B-Instruct-AWQ 2>/dev/null
echo "--- files ---"
ls -lh .models/Qwen2.5-7B-Instruct-AWQ/ 2>/dev/null | tail -6
echo "--- .part present? ---"
ls -lh .models/Qwen2.5-7B-Instruct-AWQ/*.part 2>/dev/null || echo "(no .part)"
echo "--- log tail ---"
tail -c 200 .models/logs/awq_download.log 2>/dev/null

echo
echo "=== [2] vLLM pip install ==="
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do echo "pip $pid ALIVE"; done
echo "venv: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)"
echo "pip cache: $(du -sh /mnt/d/pipcache 2>/dev/null | cut -f1)  files: $(find /mnt/d/pipcache -name '*.body' 2>/dev/null | wc -l)"
echo "--- biggest cached wheels ---"
find /mnt/d/pipcache -name '*.body' -size +50M 2>/dev/null -exec ls -lh {} \; | awk '{print $5, $9}' | tail -6
echo "--- log tail ---"
tail -c 250 .models/logs/vllm_install2.log

echo
echo "=== disk ==="
df -h / /mnt/d | tail -2
echo
echo "=== are the jobs detached (survive my session)? ==="
ps -o pid,ppid,pgid,etime,cmd -p "$(cat .models/logs/awq.pid 2>/dev/null)" 2>/dev/null || true
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do
  ps -o pid,ppid,pgid,etime,cmd -p "$pid" 2>/dev/null | tail -1
done
