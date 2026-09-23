#!/usr/bin/env bash
# Stop everything I started for the install; report the final state. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== stopping the version probe / pip ==="
pkill -f 'vllm_probe' 2>/dev/null && echo "  killed probe shells" || echo "  no probe shells"
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do
  kill "$pid" 2>/dev/null && echo "  killed pip $pid"
done
for f in .models/logs/vllm_install2.pid .models/logs/vllm_install.pid; do
  [ -f "$f" ] && { kill "$(cat "$f")" 2>/dev/null && echo "  killed wrapper in $f"; }
done
sleep 2
echo "remaining pip processes: $(pgrep -cf 'venvs/vllm/bin/pip' 2>/dev/null || echo 0)"

echo
echo "=== nothing of mine should be running now ==="
ps -eo pid,etime,cmd | grep -E 'venvs/vllm|hf_mirror_download' | grep -v grep || echo "  (clean)"

echo
echo "=== state summary ==="
echo "AWQ model:      $(du -sh .models/Qwen2.5-7B-Instruct-AWQ 2>/dev/null | cut -f1) (complete)"
echo "7B fp16 model:  $(du -sh .models/Qwen2.5-7B-Instruct 2>/dev/null | cut -f1)"
echo "0.5B model:     $(du -sh .models/Qwen2.5-0.5B-Instruct 2>/dev/null | cut -f1)"
echo "vllm venv:      $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)  (NOT usable: install stopped)"
echo "pip cache:      $(du -sh /mnt/d/pipcache 2>/dev/null | cut -f1)"
echo
echo "driver:         $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"
/usr/lib/wsl/lib/nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | sed 's/^/                /'
echo
echo "=== repo state ==="
git status --sb | head -3
echo "unpushed: $(git log --oneline origin/main..main 2>/dev/null | wc -l)"
echo
df -h / /mnt/d | tail -2
