#!/usr/bin/env bash
# Report the WSL networking situation: NAT vs mirrored, and what the host sees.
# ASCII ONLY.
set -u

echo "=== WSL addresses ==="
ip -4 addr show 2>/dev/null | grep -E 'inet |^[0-9]+:' | sed 's/^/  /'

echo
echo "=== default route ==="
ip route show default 2>/dev/null | sed 's/^/  /'

echo
echo "=== is this NAT (172.x) or mirrored (host LAN IP)? ==="
WSLIP=$(ip -4 addr show eth0 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)
echo "  eth0 address: ${WSLIP:-none}"
case "${WSLIP:-}" in
  172.1[6-9].*|172.2[0-9].*|172.3[01].*) echo "  -> looks like WSL2 NAT (not reachable from other machines without help)" ;;
  127.*) echo "  -> loopback only" ;;
  *) echo "  -> not a typical NAT range; possibly mirrored networking" ;;
esac

echo
echo "=== interfaces list (which one would NCCL pick?) ==="
ip -o link show 2>/dev/null | awk -F': ' '{print "  " $2}' | head -10

echo
echo "=== can we see the Windows host from inside? (mirrored mode check) ==="
ip route show 2>/dev/null | head -5 | sed 's/^/  /'

echo
echo "=== /etc/wsl.conf ==="
cat /etc/wsl.conf 2>/dev/null | sed 's/^/  /'

echo
echo "=== Windows-side .wslconfig (if visible via /mnt/c) ==="
for f in /mnt/c/Users/*/.wslconfig; do
  [ -f "$f" ] && { echo "  --- $f ---"; sed 's/^/    /' "$f"; }
done
echo "(nothing printed = no .wslconfig -> default NAT networking)"

echo
echo "=== nvidia-smi / GPU ==="
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo
echo "=== listen ports that matter for the cluster ==="
ss -lntp 2>/dev/null | grep -E ':(2222|6379|8000|8100|8265)\s' | sed 's/^/  /' || echo "  (none of 2222/6379/8000/8100/8265 listening)"
