#!/usr/bin/env bash
# Is vLLM 0.10.1 installed and working? ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"
LOG=.models/logs/vllm10_install.log

echo "=== process ==="
if [ -f .models/logs/vllm10.pid ] && kill -0 "$(cat .models/logs/vllm10.pid)" 2>/dev/null; then
  echo "wrapper ALIVE (pid $(cat .models/logs/vllm10.pid))"
else
  echo "wrapper: finished or gone"
fi
pgrep -af 'venvs/vllm/bin/pip' | head -3 || echo "  no pip running"

echo
echo "=== progress markers in the log ==="
grep -E '^### ' "$LOG" 2>/dev/null | tail -8
echo "downloads: $(grep -c 'Downloading' "$LOG" 2>/dev/null)  |  last line:"
tail -2 "$LOG" 2>/dev/null

echo
echo "=== venv size / disk ==="
du -sh "$VENV" 2>/dev/null
df -h / /mnt/d | tail -2

echo
echo "=== CAN WE IMPORT / IS CUDA ALIVE? ==="
if [ -x "$VENV/bin/python" ]; then
  "$VENV/bin/python" - <<'EOF' 2>&1 | tail -12
import importlib
for mod in ("torch", "vllm", "lmcache", "ray"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:8s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {mod:8s} FAILED: {type(exc).__name__}: {str(exc)[:100]}")
try:
    import torch
    print("  torch cuda build:", torch.version.cuda)
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        x = torch.randn(256, 256, device="cuda")
        print("  gpu matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
        print("  device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("  cuda check failed:", type(exc).__name__, str(exc)[:120])
EOF
fi
echo
echo "=== vllm CLI present? ==="
ls -l "$VENV/bin/vllm" 2>/dev/null || echo "  (no vllm CLI yet)"
