#!/usr/bin/env bash
# Stop the cu13 install and find the newest vLLM whose torch is cu12x.
#
# Why: the WSL driver is 566.24 = CUDA 12.7. CUDA minor-version compatibility does
# NOT cross major versions, so cu13 wheels need driver >= 580 and will fail at
# runtime ("CUDA driver version is insufficient"). cu128 needs >= 570 too, so we
# need a vLLM whose torch is cu124/cu126.
#
# `pip install --dry-run --report` resolves the tree WITHOUT downloading, so we can
# read each candidate's torch + nvidia-*cuNN pins cheaply.
#
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"
export PIP_CACHE_DIR=/mnt/d/pipcache
export PIP_DISABLE_PIP_VERSION_CHECK=1
IDX=https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== stop the running cu13 install ==="
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do
  kill "$pid" 2>/dev/null && echo "  killed pip $pid"
done
if [ -f .models/logs/vllm_install2.pid ]; then
  W=$(cat .models/logs/vllm_install2.pid)
  kill "$W" 2>/dev/null && echo "  killed wrapper $W" || true
fi
sleep 2
echo "remaining pip: $(pgrep -cf 'venvs/vllm/bin/pip' 2>/dev/null || echo 0)"

probe() {
  local ver="$1"
  local rep="/tmp/vllm_probe_${ver}.json"
  rm -f "$rep"
  timeout 180 "$VENV/bin/pip" install --dry-run --quiet \
    --index-url "$IDX" --report "$rep" "vllm==${ver}" >/dev/null 2>&1
  if [ ! -s "$rep" ]; then
    printf '  %-10s RESOLVE FAILED\n' "$ver"
    return
  fi
  "$VENV/bin/python" - "$rep" "$ver" <<'EOF'
import json, sys, re
rep, ver = sys.argv[1], sys.argv[2]
d = json.load(open(rep))
torch = ""
cuda = set()
for item in d.get("install", []):
    name = item["metadata"]["name"].lower()
    v = item["metadata"]["version"]
    if name == "torch":
        torch = v
    m = re.match(r"nvidia[_-].*?cu(\d+)", name.replace("-", "_"))
    if m:
        cuda.add(m.group(1))
cuda_s = ",".join(sorted(cuda)) or "?"
ok = torch and cuda and all(c in ("12",) for c in cuda)
print(f"  {ver:<10} torch={torch:<12} cuda_major={cuda_s:<8} {'<-- CUDA12 OK' if ok else ''}")
EOF
}

echo
echo "=== probing candidates (newest first) ==="
for v in 0.11.0 0.10.2 0.10.0 0.9.2 0.9.0 0.8.5; do probe "$v"; done

echo
echo "=== available versions (tail) ==="
timeout 120 "$VENV/bin/pip" index versions vllm --index-url "$IDX" 2>&1 | head -3
