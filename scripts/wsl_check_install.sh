#!/usr/bin/env bash
# Check the vLLM install progress and the ssh socket configuration. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "=== vLLM install progress ==="
tail -c 500 .models/logs/vllm_install.log 2>/dev/null || echo "(no log)"
echo
if [ -f .models/logs/vllm_install.pid ] && kill -0 "$(cat .models/logs/vllm_install.pid)" 2>/dev/null; then
  echo "install pid $(cat .models/logs/vllm_install.pid): ALIVE"
else
  echo "install: NOT RUNNING (finished or died)"
fi
echo "venv size: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)"
echo "venv python: $([ -x "$HOME/venvs/vllm/bin/python" ] && echo present || echo missing)"

echo
echo "=== why port 2222? ==="
echo "--- ssh.socket unit ---"
systemctl cat ssh.socket 2>/dev/null | grep -iE 'listenstream|^\[|^#' | head -10
echo "--- drop-ins ---"
ls -1 /etc/systemd/system/ssh.socket.d/ 2>/dev/null || echo "(none)"
echo "--- ssh.service active? ---"
systemctl is-active ssh 2>/dev/null || true
echo "--- ssh.socket enabled? ---"
systemctl is-enabled ssh.socket 2>/dev/null || true
echo "--- listeners ---"
ss -lntp 2>/dev/null | grep -E ':(22|2222)\s' || echo "(none on 22/2222)"
echo "--- sshd_config Port lines ---"
grep -rn '^Port' /etc/ssh/sshd_config 2>/dev/null || echo "(no explicit Port = default 22)"
echo "--- /etc/wsl.conf ---"
cat /etc/wsl.conf 2>/dev/null || echo "(none)"
