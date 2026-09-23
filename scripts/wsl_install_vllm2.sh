#!/usr/bin/env bash
# Restart the vLLM install with a WORKING index and unbuffered logging.
#
# Why: the machine's pip config points at http://mirrors.aliyun.com/pypi/simple/
# with install-scoped trusted-host, which newer pip ignores -> the install stalls.
# Pick the first mirror that actually serves /simple/vllm/ over HTTPS.
#
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
mkdir -p .models/logs
VENV="${VENV:-$HOME/venvs/vllm}"
export PIP_CACHE_DIR=/mnt/d/pipcache
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
LOG=.models/logs/vllm_install2.log
mkdir -p "$PIP_CACHE_DIR"

echo "=== stop the stuck attempt ==="
pkill -f "venvs/vllm/bin/pip" 2>/dev/null && echo "killed pip" || echo "no pip to kill"
if [ -f .models/logs/vllm_install.pid ]; then
  OLDPID=$(cat .models/logs/vllm_install.pid)
  kill "$OLDPID" 2>/dev/null && echo "killed old wrapper $OLDPID" || true
fi
sleep 3

echo
echo "=== choosing an https index ==="
BEST=""
for IDX in \
  "https://pypi.tuna.tsinghua.edu.cn/simple" \
  "https://mirrors.aliyun.com/pypi/simple/" \
  "https://mirrors.ustc.edu.cn/pypi/simple/" \
  "https://pypi.org/simple" ; do
  CODE=$(timeout 25 curl -s -o /dev/null -w '%{http_code}' "${IDX%/}/vllm/" 2>/dev/null || echo 000)
  printf '  %-45s %s\n' "$IDX" "$CODE"
  if [ "$CODE" = "200" ] && [ -z "$BEST" ]; then BEST="$IDX"; fi
done
[ -z "$BEST" ] && { echo "NO_WORKING_INDEX"; exit 1; }
echo "chosen: $BEST"

echo
echo "=== launching install in background (unbuffered log: $LOG) ==="
cat > .models/logs/run_vllm_install.sh <<EOF
#!/usr/bin/env bash
set -u
VENV="$VENV"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
IDX="$BEST"
LOG="$HOME/pair/$LOG"
cd "$HOME/pair" || exit 1
{
  echo "### pip upgrade"
  "\$VENV/bin/pip" install --index-url "\$IDX" --timeout 60 --retries 5 -U pip setuptools wheel
  echo "### vllm"
  "\$VENV/bin/pip" install --index-url "\$IDX" --timeout 60 --retries 5 vllm
  echo "### vllm rc=\$?"
  echo "### lmcache"
  "\$VENV/bin/pip" install --index-url "\$IDX" --timeout 60 --retries 5 lmcache
  echo "### lmcache rc=\$?"
  echo "### versions"
  "\$VENV/bin/python" -c "import torch,vllm;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('vllm',vllm.__version__)"
  echo "### DONE"
} >> "\$LOG" 2>&1
EOF
chmod +x .models/logs/run_vllm_install.sh
nohup bash .models/logs/run_vllm_install.sh > /dev/null 2>&1 &
echo $! > .models/logs/vllm_install2.pid
echo "started pid $(cat .models/logs/vllm_install2.pid)"

sleep 45
echo
echo "=== log after 45s ==="
tail -n 20 "$LOG"
echo
echo "venv size: $(du -sh "$VENV" 2>/dev/null | cut -f1)"
