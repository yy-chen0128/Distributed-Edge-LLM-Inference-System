#!/usr/bin/env bash
# V1a: verify LMCache really imports (incl. its C extension) and that a real
# vLLM 0.10.1 server answers a request. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"
PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct
PORT=8000
LOG=.models/logs/vllm_v1a_server.log

echo "=== 1. LMCache import detail ==="
"$PY" - <<'EOF' 2>&1 | tail -14
import traceback
try:
    import lmcache
    print("  lmcache file:", lmcache.__file__)
except Exception:
    print("  import lmcache FAILED"); traceback.print_exc(); raise SystemExit(0)
for attr in ("__version__", "VERSION"):
    if hasattr(lmcache, attr):
        print(f"  lmcache.{attr} =", getattr(lmcache, attr))
try:
    from lmcache._version import __version__ as v
    print("  lmcache._version.__version__ =", v)
except Exception as exc:
    print("  lmcache._version:", type(exc).__name__, exc)
# the C extension is the risky part (built against some torch ABI)
try:
    import lmcache.c_ops as c
    print("  lmcache.c_ops OK:", [a for a in dir(c) if not a.startswith('_')][:5])
except Exception as exc:
    print("  lmcache.c_ops FAILED:", type(exc).__name__, str(exc)[:160])
for mod in ("lmcache.integration.vllm.lmcache_connector_v1",
            "lmcache.integration.vllm.lmcache_connector"):
    try:
        __import__(mod)
        print(f"  {mod}: OK")
    except Exception as exc:
        print(f"  {mod}: {type(exc).__name__} {str(exc)[:90]}")
try:
    import vllm
    from vllm.config import KVTransferConfig
    print("  vllm KVTransferConfig import OK")
except Exception as exc:
    print("  vllm KVTransferConfig:", type(exc).__name__, str(exc)[:90])
EOF

echo
echo "=== 2. start vLLM 0.10.1 (plain, no LMCache) ==="
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
  --max-model-len 8192 --gpu-memory-utilization 0.55 \
  > "$LOG" 2>&1 &
SPID=$!
echo "server pid $SPID; waiting for ready..."

ready=0
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" != "1" ]; then
  echo "NOT READY after $((i*5))s; last 25 log lines:"
  tail -25 "$LOG"
  kill "$SPID" 2>/dev/null
  exit 1
fi
echo "READY after ~$((i*5))s"
echo "--- server startup banner ---"
grep -iE 'Starting vLLM|version|model|pipeline|GPU|graph capture|init engine' "$LOG" | head -12
echo
echo "--- /v1/models ---"
curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 300
echo
echo
echo "=== 3. one real request (shared long prefix x3) ==="
"$PY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$PORT" \
  --prefix-tokens 3000 --requests 3 --max-tokens 16 --label V1a-plain 2>&1 | tail -8

echo
echo "=== 4. stop server ==="
kill "$SPID" 2>/dev/null
sleep 5
pkill -f 'vllm serve' 2>/dev/null
echo done
