#!/usr/bin/env bash
# Run the project test suite inside WSL with the GPU venv.
# Fed to: ssh ... "tr -d '\r' | bash -s"
#
# ASCII ONLY -- both the script body AND any path literal it contains.
# PowerShell re-encodes stdin/args using the Windows console codepage (GBK here),
# so a non-ASCII path like /mnt/d/Newproject/<CJK>/project arrives as mojibake.
# Resolve it with find/glob instead of typing it.
set -u
PY="$HOME/venvs/pair/bin/python"

PROJ="$(find /mnt/d/Newproject -maxdepth 2 -type d -name project 2>/dev/null | head -1)"
if [ -z "$PROJ" ]; then echo "PROJECT_NOT_FOUND"; exit 1; fi
echo "[path] resolved project = $PROJ"
ln -sfn "$PROJ" "$HOME/pair"
echo "[path] ascii alias       = $HOME/pair -> $PROJ"

cd "$PROJ" || exit 1
export PYTHONPATH=.
echo "[env] python = $PY"
"$PY" -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "[model]"
ls -d .models/*/ 2>/dev/null | head -5

echo "[pytest] starting"
"$PY" -m pytest edge_llm_scheduler/tests/ -q -p no:cacheprovider 2>&1 | tail -6
