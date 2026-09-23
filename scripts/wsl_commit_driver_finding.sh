#!/usr/bin/env bash
# Commit the driver/CUDA compatibility finding. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
git add -A
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: record the driver/CUDA constraint that blocks vLLM+LMCache as installed

Measured 2026-09-23: vLLM 0.30.0 and LMCache 0.5.5 both resolve to torch 2.13/2.14
with cu130 wheels, but this machine's WSL driver is 566.24 (CUDA 12.7). CUDA
major versions are not backward compatible, so after installing:
  torch 2.14.0+cu130, cuda available: False
  UserWarning: The NVIDIA driver on your system is too old (found version 12070)
The existing ~/venvs/pair works fine on the same driver with torch 2.6.0+cu124,
which proves it is a wheel/driver mismatch, not a broken GPU or WSL setup.

Two ways out, to be decided before installing again:
1. upgrade the Windows-side NVIDIA driver to >= 580 (WSL's CUDA support comes from
   the Windows driver) and keep vLLM 0.30 + LMCache 0.5.5;
2. pin an older combination whose torch is cu124/cu126 (vLLM <= 0.9.x plus the
   matching LMCache), verifying each pin.

All four machines must satisfy the same constraint, so the interconnect checklist
now asks for each machine's driver version and explains why. The plan doc gains a
fifth hard constraint (numbered 0) with the measured compatibility table, and the
install is stopped until the choice is made.

Note: the install was interrupted after ~33 minutes; D: now has 24 GB free because
the AWQ model (5.2 GB), the fp16 7B (15 GB) and the pip cache (7.5 GB, mostly cu13
wheels that will not be used) are on it. No further downloads are running.
EOF
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -F "$MSGFILE" >/dev/null
rm -f "$MSGFILE"

echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -3
git log --oneline -3
echo
echo "=== confirm nothing is running ==="
ps -eo pid,cmd 2>/dev/null | grep -E 'venvs/vllm|hf_mirror|pip install' | grep -v grep || echo "  (clean: no install, no download)"
