#!/usr/bin/env bash
# 通过 ssh 在 WSL 里跑的环境探测（由 Windows 侧喂给 bash -s）
export PATH="$PATH:/usr/lib/wsl/lib:/usr/local/cuda/bin"
echo "--- 用户/系统 ---"; whoami; . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"; uname -r
echo "--- GPU（nvidia-smi 绝对路径）---"
if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
else
  echo "!! /usr/lib/wsl/lib/nvidia-smi 不存在 → GPU 直通可能没配好"
  ls -l /dev/dxg 2>/dev/null || echo "!! 也没有 /dev/dxg"
fi
echo "--- python / venv ---"
python3 -V
ls -d "$HOME"/venvs/* 2>/dev/null || echo "(无 ~/venvs)"
for p in "$HOME"/venvs/*/bin/python; do
  [ -x "$p" ] && printf '%s -> ' "$p" && "$p" -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1
done
echo "--- 是否有 pip 在下载 ---"
pgrep -af "pip install" | head -3 || echo "(无)"
echo "--- 项目与模型 ---"
ls -d "$HOME"/pair 2>/dev/null || echo "(无 ~/pair)"
ls -d "$HOME"/models/* 2>/dev/null | head -5 || echo "(无 ~/models)"
echo "--- 磁盘 ---"; df -h / | tail -1; df -h /mnt/d 2>/dev/null | tail -1
echo "--- sshd ---"; ss -tlnp 2>/dev/null | grep -i ssh || echo "(ss 无 ssh 行；可能是 ssh.socket 激活)"
systemctl is-active ssh 2>/dev/null; systemctl is-active ssh.socket 2>/dev/null
echo "--- wsl.conf ---"; cat /etc/wsl.conf 2>/dev/null || echo "(无)"
echo "--- 内存 ---"; free -h | head -2
