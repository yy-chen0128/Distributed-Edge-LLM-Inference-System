#!/usr/bin/env bash
# WSL-side env check. Fed to: ssh ... "tr -d '\r' | bash -s"
# KEEP THIS FILE ASCII-ONLY. PowerShell pipes it to the native ssh process using the
# console codepage (GBK here), so non-ASCII bytes get RE-ENCODED on the way and can
# corrupt/consume later lines of the script (seen: an `export PATH=...` line silently
# disappeared because a Chinese comment above it was re-encoded).
export PATH="$PATH:/usr/lib/wsl/lib"
NVSMI=/usr/lib/wsl/lib/nvidia-smi
PY="$HOME/venvs/pair/bin/python"

echo "[pip]"
if pgrep -f "pip install" >/dev/null 2>&1; then
  echo "STILL_RUNNING: $(pgrep -af 'pip install' | head -1 | cut -c1-120)"
else
  echo "DONE"
fi

echo "[python]"
echo "path=$PATH"
command -v nvidia-smi || echo "nvidia-smi NOT in PATH (use absolute path)"

echo "[torch]"
"$PY" - <<'PYCODE' 2>&1 | tail -4
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__)
PYCODE

echo "[gpu]"
"$NVSMI" --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

echo "[misc]"
echo "pytest: $("$PY" -c 'import pytest;print(pytest.__version__)' 2>&1 | tail -1)"
echo "home: $HOME"
df -h / | tail -1
