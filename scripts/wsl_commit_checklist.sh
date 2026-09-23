#!/usr/bin/env bash
# Commit the verification checklist. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
git add -A
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: verification checklist with explicit single-machine vs real-hardware split

The plan doc's V1..V4 table gains V2b (a local Ray cluster with PP over Ray, which
exercises the exact code path the four machines will use -- the riskiest step, and
doable on one machine) plus a 12-item checklist that marks each item as doable
single-machine or requiring real hardware.

Items doable here: vLLM single instance, LMCache hit, LMCache surviving a restart
(decides the tier-1 bubble), PP=4 on one oversubscribed GPU (mechanism), local Ray
PP, vLLM startup/load time (the first component of the tier-0 bubble), AWQ int4 7B
loading and per-rank memory (decides whether 4 GB machines can take part), the
gateway end-to-end drill (kill, restart, replay, client output complete and not
duplicated) and replay fidelity.

Items that cannot be substituted: TOTP cross-machine cost (4 machines PP=4 versus
single-machine PP=1) -- the decisive number for whether cross-machine is worth it;
per-machine memory; and real WiFi RTT/throughput/jitter/loss.

Also started downloading Qwen2.5-7B-Instruct-AWQ in the background (2 shards,
~7.4 GB) because 4 GB GPUs cannot hold 7 fp16 layers (3.26 GB) but can hold 7 int4
layers (~0.9 GB).
EOF
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -F "$MSGFILE" >/dev/null
rm -f "$MSGFILE"

echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -3

echo
echo "=== downloads ==="
echo "vLLM venv: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)  pip cache: $(du -sh /mnt/d/pipcache 2>/dev/null | cut -f1)"
pgrep -cf 'venvs/vllm/bin/pip' | sed 's/^/  pip processes: /'
echo "AWQ: $(du -sh .models/Qwen2.5-7B-Instruct-AWQ 2>/dev/null | cut -f1)  $(tail -c 90 .models/logs/awq_download.log 2>/dev/null)"
tail -c 150 .models/logs/vllm_install2.log
echo
df -h / /mnt/d | tail -2
