#!/usr/bin/env bash
# Commit the interconnect doc + environment-setup vLLM route. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

git add -A
echo "=== staged ==="
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: four-machine interconnect playbook; vLLM route in environment-setup

New doc docs/deploy/four-machine-interconnect.md, answering "how do we connect
them, what do you need from me, and is the per-machine setup clear":
1. what to provide: a four-row table (OS, GPU, VRAM, LAN IP, link type, disk,
   whether I can ssh) plus three go/no-go confirmations (machines can ping each
   other, ports 6379/8000/8100 + 10002-10100 allowed, each WSL in mirrored mode),
   plus collect_env.py output and the pairwise link measurements.
2. network feasibility, quantified: PP sends one activation per token per hop, so
   a 7B pipeline moves 7,168 B/token/hop and 21.5 KB/token over three hops, i.e.
   3.4 Mbit/s at 20 tok/s (0.9 Mbit/s for 0.5B). Bandwidth is never the problem;
   latency is, because every token crosses three hops serially. This matches
   TPI-LLM's finding that the allreduce bottleneck is link latency, not bandwidth.
3. per-machine deltas vs the self-built route: separate ~/venvs/vllm (vLLM 0.30
   pulls torch 2.13), explicit https pip index, PIP_CACHE_DIR on /mnt/d, Ray,
   model distribution options, port list, NCCL/GLOO_SOCKET_IFNAME, clock sync.
4. startup order (ray head/workers -> vllm serve PP=4 via ray -> gateway), an
   8-item acceptance checklist including the decisive one (per-token cost with 4
   machines vs PP=1 on one machine), the node-departure drill for tier 0 + tier 1,
   and a troubleshooting table.

Two machine facts measured today that the playbook depends on:
- The WSL distro is in MIRRORED networking mode (the Windows-side .wslconfig has
  networkingMode=mirrored), and the WSL interface for the WLAN carries the same
  address as the Windows WLAN adapter. Without mirrored mode a default-NAT distro is
  unreachable from the other machines, so Ray/vLLM PP could never start. Each of
  the four machines needs this.
- The host is on a campus network (10.20.x.x/17, gateway 10.20.0.1). Campus WiFi
  often enables client isolation, which would block the cluster outright, so
  "can the four machines ping each other" is the first go/no-go check. Also, WSL
  exposes six interfaces here and the WLAN maps to eth3, so NCCL would very likely
  pick the wrong one -- NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME must be set per machine.

environment-setup.md:
- scope note at the top splitting the two routes (vLLM is the main line now;
  sections 4-6 belong to the self-built engine route) so a classmate knows which
  half to follow;
- new 2.6 with the vLLM/LMCache environment: separate venv, the pip-index trap that
  stalled the install at 3 MB (/etc/pip.conf points at an http Aliyun mirror whose
  trusted-host scope current pip ignores), PIP_CACHE_DIR on /mnt/d, acceptance
  commands, the 6-9 GB venv size warning, and the model tier table that follows
  from the smallest GPU (4 GB machines must use AWQ/GPTQ int4).
EOF
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -F "$MSGFILE" >/dev/null
rm -f "$MSGFILE"

echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -3

echo
echo "=== vLLM install ==="
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do echo "  pip $pid alive"; done
echo "venv: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)  cache: $(du -sh /mnt/d/pipcache 2>/dev/null | cut -f1)"
tail -c 220 .models/logs/vllm_install2.log
