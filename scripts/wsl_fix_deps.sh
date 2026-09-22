#!/usr/bin/env bash
# Install optional deps into the WSL venv and re-run the suite.
# ASCII ONLY (see wsl_run_tests.sh header for why).
set -u
PY="$HOME/venvs/pair/bin/python"
PROJ="$(find /mnt/d/Newproject -maxdepth 2 -type d -name project 2>/dev/null | head -1)"
[ -z "$PROJ" ] && { echo "PROJECT_NOT_FOUND"; exit 1; }

echo "[pip] installing aiohttp psutil"
"$PY" -m pip install -q aiohttp psutil 2>&1 | tail -3
"$PY" -c "import aiohttp, psutil; print('[pip] aiohttp', aiohttp.__version__, '| psutil', psutil.__version__)"

cd "$PROJ" || exit 1
export PYTHONPATH=.
echo "[pytest] re-run"
"$PY" -m pytest edge_llm_scheduler/tests/ -q -p no:cacheprovider 2>&1 | tail -4
