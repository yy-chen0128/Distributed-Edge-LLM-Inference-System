#!/usr/bin/env bash
# Print the things the other machine needs from this one:
#   - this machine's LAN IP, GPU facts, sshd port
#   - the public key that must be authorised on the peer (if the agent is to have
#     direct access), plus the exact command to authorise it
#   - a ready-to-paste readiness command for the peer
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

echo "############ THIS MACHINE (node A candidate) ############"
LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)
IFACE=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'dev \K[^ ]+' | head -1)
echo "  hostname   : $(hostname)"
echo "  LAN IP     : ${LAN_IP:-unknown}"
echo "  LAN iface  : ${IFACE:-unknown}     (use this for NCCL_SOCKET_IFNAME)"
echo "  all addrs  : $(ip -4 addr show 2>/dev/null | grep -oP 'inet \K[0-9.]+' | tr '\n' ' ')"
echo "  sshd       : $(ss -lnt 2>/dev/null | grep -oP 'LISTEN.*?:\K[0-9]+(?= )' | sort -u | tr '\n' ' ')"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/  gpu        : /'
/usr/lib/wsl/lib/nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | sed 's/^/  driver cuda: /'
echo "  venv vllm  : $("$HOME/venvs/vllm/bin/python" -c "import torch,vllm;print('torch',torch.__version__,'vllm',vllm.__version__,'cuda',torch.cuda.is_available())" 2>/dev/null)"

echo
echo "############ PUBLIC KEY (for the peer, if I should control it directly) ############"
if [ -f .ssh/agent_ed25519.pub ]; then
  cat .ssh/agent_ed25519.pub
else
  echo "  (no .ssh/agent_ed25519.pub in the repo; the bridge key lives there)"
fi
echo
echo "  To authorise it on the PEER (run there, in WSL):"
echo "    mkdir -p ~/.ssh && chmod 700 ~/.ssh"
echo "    echo '<the line above>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
echo "  Then from here:  ssh -i .ssh/agent_ed25519 -p <peer-sshd-port> <peer-user>@<peer-ip>"

echo
echo "############ COMMAND FOR THE PEER (run in ITS WSL) ############"
cat <<'EOS'
  cd <repo>
  python scripts/check_node_readiness.py --peer <THIS machine's LAN IP> \
         --json-out node_readiness_$(hostname).json
  # then send back the "### SEND THESE BACK TO THE CONTROLLER ###" block
EOS

echo
echo "############ can this machine see anything at all? ############"
echo "  default route: $(ip route show default 2>/dev/null | head -1)"
