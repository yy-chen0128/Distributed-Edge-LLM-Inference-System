#!/usr/bin/env bash
# Wait for the vLLM install to finish, then report versions and any error.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
LOG=.models/logs/vllm_install2.log

echo "=== waiting for pip to finish (max ~40 min) ==="
for i in $(seq 1 240); do
  if ! pgrep -f 'venvs/vllm/bin/pip' >/dev/null 2>&1; then
    echo "pip finished after ~$((i * 10))s"
    break
  fi
  if [ $((i % 12)) -eq 0 ]; then
    echo "  [$((i * 10))s] venv=$(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1) $(pgrep -cf 'venvs/vllm/bin/pip') pip(s)"
  fi
  sleep 10
done

echo
echo "=== log tail ==="
tail -n 25 "$LOG" 2>/dev/null

echo
echo "=== DONE marker present? ==="
grep -c '^### DONE' "$LOG" 2>/dev/null || true

echo
echo "=== versions ==="
"$HOME/venvs/vllm/bin/python" - <<'EOF' 2>&1 | tail -8
import importlib
for mod in ("torch", "vllm", "lmcache"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:8s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {mod:8s} FAILED: {type(exc).__name__}: {str(exc)[:100]}")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"  cuda check failed: {type(exc).__name__}")
EOF

echo
echo "=== vllm CLI present? ==="
ls -l "$HOME/venvs/vllm/bin/vllm" 2>/dev/null || echo "  (no vllm CLI)"
echo
echo "=== disk ==="
df -h / /mnt/d | tail -3
