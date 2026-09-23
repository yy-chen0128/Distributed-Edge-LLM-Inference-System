#!/usr/bin/env bash
# Install vLLM + LMCache into a SEPARATE venv (do not touch ~/venvs/pair).
#
# Why separate: vLLM pulls its own torch build; installing into the existing venv
# would replace torch 2.6.0+cu124 that the layered-engine experiments depend on.
# Why PIP_CACHE_DIR on /mnt/d: the WSL ext4.vhdx only grows, never shrinks.
#
# ASCII ONLY.
set -u
VENV="${VENV:-$HOME/venvs/vllm}"
export PIP_CACHE_DIR=/mnt/d/pipcache
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== disk before ==="
df -h / /mnt/d | tail -3
mkdir -p "$PIP_CACHE_DIR"

if [ ! -x "$VENV/bin/python" ]; then
  echo "=== creating venv $VENV ==="
  python3 -m venv "$VENV" || { echo "VENV_FAILED"; exit 1; }
fi
"$VENV/bin/python" -m pip install -q -U pip setuptools wheel || exit 1

echo
echo "=== installing vllm (this pulls its own torch; expect several GB) ==="
"$VENV/bin/pip" install vllm 2>&1 | tail -8
VLLM_RC=${PIPESTATUS[0]}

echo
echo "=== installing lmcache ==="
"$VENV/bin/pip" install lmcache 2>&1 | tail -6

echo
echo "=== versions ==="
"$VENV/bin/python" - <<'EOF'
import importlib
for mod in ("torch", "vllm", "lmcache"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:8s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {mod:8s} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()} "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
except Exception as exc:
    print(f"  cuda check failed: {type(exc).__name__}")
EOF

echo
echo "=== disk after ==="
df -h / /mnt/d | tail -3
echo "vllm install exit code: $VLLM_RC"
