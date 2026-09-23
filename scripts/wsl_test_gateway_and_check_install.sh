#!/usr/bin/env bash
# Run the gateway unit tests, install the gateway's web deps, and report the
# vLLM install progress. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
PY="$HOME/venvs/pair/bin/python"

echo "=== gateway pure-logic tests ==="
PYTHONPATH=. "$PY" -m pytest edge_llm_scheduler/tests/test_gateway_replay.py -q -p no:cacheprovider 2>&1 | tail -6

echo
echo "=== full suite (regression) ==="
PYTHONPATH=. "$PY" -m pytest edge_llm_scheduler/tests -q -p no:cacheprovider 2>&1 | tail -3

echo
echo "=== install gateway web deps into ~/venvs/pair (fastapi/uvicorn/httpx) ==="
export PIP_CACHE_DIR=/mnt/d/pipcache
"$PY" -m pip install -q --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  fastapi uvicorn httpx 2>&1 | tail -3
"$PY" - <<'EOF'
for mod in ("fastapi", "uvicorn", "httpx"):
    try:
        m = __import__(mod)
        print(f"  {mod:8s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {mod:8s} FAILED {type(exc).__name__}")
EOF

echo
echo "=== vLLM install progress ==="
if [ -f .models/logs/vllm_install2.pid ] && kill -0 "$(cat .models/logs/vllm_install2.pid)" 2>/dev/null; then
  echo "pid $(cat .models/logs/vllm_install2.pid): ALIVE"
else
  echo "install wrapper: NOT RUNNING"
fi
pgrep -af 'venvs/vllm' | head -3
echo "venv size: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)"
echo "--- log tail ---"
tail -c 600 .models/logs/vllm_install2.log 2>/dev/null
echo
echo "--- disk ---"
df -h / /mnt/d | tail -3
