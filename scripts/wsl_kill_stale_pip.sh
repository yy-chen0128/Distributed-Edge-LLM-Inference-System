#!/usr/bin/env bash
# Kill the stale pip from the first (broken-index) install attempt so only the
# Tsinghua-index install proceeds. Racing installs into one venv corrupt it.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== pip processes with start time ==="
ps -eo pid,lstart,etime,cmd | grep -E 'venvs/vllm/bin/pip' | grep -v grep

echo
echo "=== identify the stale one (no --index-url = the old script) ==="
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do
  CMD=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  case "$CMD" in
    *"--index-url"*) echo "  KEEP  $pid : $CMD" ;;
    *) echo "  KILL  $pid : $CMD" ;;
  esac
done

echo
echo "=== killing stale pips (and any wrapper from the first script) ==="
pkill -f 'run_vllm_install_old' 2>/dev/null || true
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do
  CMD=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  case "$CMD" in
    *"--index-url"*) : ;;
    *) kill "$pid" 2>/dev/null && echo "  killed $pid" ;;
  esac
done
# also the very first wrapper (holds the old pipeline open)
if [ -f .models/logs/vllm_install.pid ]; then
  OLD=$(cat .models/logs/vllm_install.pid)
  kill "$OLD" 2>/dev/null && echo "  killed old wrapper $OLD" || echo "  old wrapper already gone"
fi
sleep 3

echo
echo "=== remaining pip processes (should be exactly one, with --index-url) ==="
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do
  echo "  $pid : $(tr '\0' ' ' < /proc/$pid/cmdline)"
done
echo "(none listed = install already moved past the download phase)"

echo
echo "=== venv size + install log tail ==="
du -sh "$HOME/venvs/vllm" 2>/dev/null
tail -c 400 .models/logs/vllm_install2.log
