#!/usr/bin/env bash
# Does the WSL driver support the CUDA major version that vLLM's torch wants?
# vLLM 0.30 -> torch 2.13 -> nvidia-*-cu13 wheels, which need a CUDA 13 capable driver.
# ASCII ONLY.
set -u

echo "=== driver + CUDA version reported by the WSL driver ==="
/usr/lib/wsl/lib/nvidia-smi 2>/dev/null | head -4
echo
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader 2>/dev/null

echo
echo "=== what the existing pair venv (torch 2.6.0+cu124) says ==="
"$HOME/venvs/pair/bin/python" - <<'EOF' 2>&1 | tail -5
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("torch built for CUDA", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
EOF

echo
echo "=== cached nvidia wheels so far (which CUDA major?) ==="
find /mnt/d/pipcache -name '*.body' -size +100M 2>/dev/null | wc -l
echo "nvidia cu12 bodies: $(find /mnt/d/pipcache -name '*.body' 2>/dev/null | wc -l) total (names are hashed; check the log instead)"
grep -o 'nvidia_[a-z_]*-cu[0-9]*' .models/logs/vllm_install2.log 2>/dev/null | sort -u | head -12

echo
echo "=== pip version in the vllm venv (does it support resume-retries?) ==="
"$HOME/venvs/vllm/bin/pip" --version
"$HOME/venvs/vllm/bin/pip" install --help 2>/dev/null | grep -i -A1 'resume' | head -6
