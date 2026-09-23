#!/usr/bin/env bash
# Diagnose why the vLLM install stopped. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

PIDF=.models/logs/vllm_install.pid
if [ -f "$PIDF" ]; then
  P=$(cat "$PIDF")
  echo "=== pid $P ==="
  ps -p "$P" -o pid,ppid,stat,etime,cmd 2>/dev/null || echo "  (not running)"
fi

echo
echo "=== all python processes owned by us ==="
pgrep -a -u "$(id -u)" python 2>/dev/null | head -10 || echo "  (none)"

echo
echo "=== anything referencing venvs/vllm ==="
pgrep -af 'venvs/vllm' 2>/dev/null | head -5 || echo "  (none)"

echo
echo "=== FULL install log ==="
cat .models/logs/vllm_install.log

echo
echo "=== network reachability to PyPI ==="
timeout 20 curl -sS -o /dev/null -w 'pypi.org: HTTP %{http_code} in %{time_total}s\n' https://pypi.org/simple/vllm/ 2>&1 | tail -2
timeout 20 curl -sS -o /dev/null -w 'files.pythonhosted.org: HTTP %{http_code}\n' https://files.pythonhosted.org/ 2>&1 | tail -1

echo
echo "=== pip config / index ==="
"$HOME/venvs/vllm/bin/pip" config list 2>/dev/null || echo "(no pip config)"
env | grep -i -E 'pip|proxy|index' || echo "(no pip/proxy env)"

echo
echo "=== can pip even talk to the index? (dry, fast) ==="
timeout 60 "$HOME/venvs/vllm/bin/pip" index versions vllm 2>&1 | head -5
