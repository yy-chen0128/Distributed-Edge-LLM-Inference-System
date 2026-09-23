#!/usr/bin/env bash
# Push the two local commits, then start the vLLM install in the background.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
mkdir -p .models/logs

echo "=== pending commits before push ==="
git log --oneline origin/main..main 2>/dev/null

echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -3

echo
echo "=== state ==="
git status -sb | head -2
git log --oneline -2

echo
echo "=== start vLLM + LMCache install (background) ==="
if [ -f .models/logs/vllm_install.pid ] && kill -0 "$(cat .models/logs/vllm_install.pid)" 2>/dev/null; then
  echo "already running: pid $(cat .models/logs/vllm_install.pid)"
else
  nohup bash scripts/wsl_install_vllm.sh > .models/logs/vllm_install.log 2>&1 &
  echo $! > .models/logs/vllm_install.pid
  echo "started: pid $(cat .models/logs/vllm_install.pid)"
fi
sleep 25
echo
echo "--- install log ---"
head -25 .models/logs/vllm_install.log
echo
echo "--- disk ---"
df -h / /mnt/d | tail -3
