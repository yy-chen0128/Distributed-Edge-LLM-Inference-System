#!/usr/bin/env bash
# Kill every leftover probe/pip process I started. ASCII ONLY.
set -u
pkill -f 'vllm_probe' 2>/dev/null || true
pkill -f 'dry-run' 2>/dev/null || true
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do kill "$pid" 2>/dev/null || true; done
sleep 2
echo "=== leftovers ==="
ps -eo pid,etime,cmd 2>/dev/null | grep -E 'venvs/vllm|vllm_probe|dry-run' | grep -v grep || echo "ALL_STOPPED"
echo
echo "=== disk (D: got tight: AWQ + 7B fp16 + pip cache) ==="
df -h / /mnt/d | tail -2
echo "pip cache: $(du -sh /mnt/d/pipcache 2>/dev/null | cut -f1) (mostly cu13 wheels we will not use)"
echo "vllm venv: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1) (partial, cu13 -> unusable, will be recreated)"
echo "models: $(du -sh "$HOME/pair/.models" 2>/dev/null | cut -f1) total"
