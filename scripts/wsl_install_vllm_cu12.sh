#!/usr/bin/env bash
# Rebuild the vLLM venv with the CUDA-12-compatible combination.
#
# Decision (measured, see docs/deploy/vllm-lmcache-4node-plan.md section 0):
#   driver 566.24 = CUDA 12.7  ->  cannot run cu128/cu130 wheels
#   vLLM 0.10.1 -> torch 2.7.1 (cu126)      <-- newest version that works
#   LMCache 0.3.7 -> torch unpinned by design ("let vLLM be deliberate for them")
# Install order matters: vLLM first (it pins torch), then LMCache pinned to the
# same torch so pip cannot drag torch forward.
#
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="${VENV:-$HOME/venvs/vllm}"
export PIP_CACHE_DIR=/mnt/d/pipcache
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
LOG=.models/logs/vllm10_install.log
mkdir -p .models/logs "$PIP_CACHE_DIR"

echo "=== disk before ===" | tee "$LOG"
df -h / /mnt/d | tail -2 | tee -a "$LOG"

echo "### rebuild venv (clearing the cu13 install)" | tee -a "$LOG"
python3 -m venv --clear "$VENV" >>"$LOG" 2>&1 && echo "  venv cleared+recreated" | tee -a "$LOG"
"$VENV/bin/pip" install -q -i "$IDX" -U pip setuptools wheel >>"$LOG" 2>&1

cat > .models/logs/run_vllm10.sh <<EOF
#!/usr/bin/env bash
set -u
VENV="$VENV"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
IDX="$IDX"
LOG="$HOME/pair/$LOG"
cd "$HOME/pair" || exit 1
{
  echo "### installing vllm==0.10.1 (expects torch 2.7.1 + nvidia cu126)"
  "\$VENV/bin/pip" install -i "\$IDX" --timeout 60 --retries 5 "vllm==0.10.1"
  echo "### vllm rc=\$?"
  "\$VENV/bin/python" -c "import torch;print('after vllm: torch',torch.__version__,'cuda',torch.cuda.is_available())"
  echo "### installing lmcache==0.3.7 with torch pinned so pip cannot upgrade it"
  "\$VENV/bin/pip" install -i "\$IDX" --timeout 60 --retries 5 \\
      "torch==2.7.1" "lmcache==0.3.7"
  echo "### lmcache rc=\$?"
  echo "### final versions"
  "\$VENV/bin/python" - <<'PY'
import importlib
for mod in ("torch", "vllm", "lmcache", "ray"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:8s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {mod:8s} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
try:
    import torch
    print("  cuda available:", torch.cuda.is_available())
    print("  torch cuda build:", torch.version.cuda)
    if torch.cuda.is_available():
        print("  device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("  cuda check failed:", type(exc).__name__, exc)
PY
  echo "### disk after"
  df -h / /mnt/d | tail -2
  echo "### DONE"
} >> "\$LOG" 2>&1
EOF
chmod +x .models/logs/run_vllm10.sh
nohup bash .models/logs/run_vllm10.sh > /dev/null 2>&1 &
echo $! > .models/logs/vllm10.pid
echo "install started: pid $(cat .models/logs/vllm10.pid)"

sleep 50
echo
echo "=== log after 50s ==="
tail -n 12 "$LOG"
echo
echo "venv: $(du -sh "$VENV" 2>/dev/null | cut -f1)"
