#!/usr/bin/env bash
# Commit the gateway stall-timeout fix + the drill results. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
git add -A
git diff --cached --stat

git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -F - <<'MSG'
fix+docs: gateway stall detection; tier-0/tier-1 drill works end to end

Gateway bug found by the drill and fixed: the streaming proxy used
httpx.AsyncClient(timeout=None). When the upstream process is killed mid-stream the
SSE connection neither errors nor closes, so the gateway waited forever and the
recovery path never ran (measured: hung for 10 minutes with failed=0). Added
--stall-timeout (default 30 s) as the httpx read timeout: no bytes for that long
means the upstream is gone. This is a precondition for tier 0 + tier 1 to work at
all, not an optimisation.

Drill result (PP=1 suffices for this): a 400-token streaming request through the
gateway, with vLLM killed and restarted 3 s in.
  client: 79.6 s total, 301 chunks before the kill, 14 after, 1624 chars, no error
  gateway: failed=1, replayed=1, replay_exact=0, replay_diverged=1,
           reason="upstream recovered"
  restart: vLLM ready again after ~75 s
So the mechanism works: the client's stream survives the server dying, the gateway
waits for the rebuild, replays, and the client ends with a complete stream and no
error.

Two measured findings recorded in the plan doc:
1. replay fidelity failed (replay_diverged=1). Replaying the generated TEXT
   re-tokenises it and the boundary is not equivalent to the original token
   sequence, so the continuation drifted to another topic. Exact replay needs the
   token level (prompt_token_ids, or the control plane holding token ids rather
   than text). The replay_diverged counter is the instrument for this and turns a
   stated risk into a number per experiment. User-visible consequence, which must
   be declared in the experiment protocol: after an interruption the answer may
   differ.
2. vLLM's PP>1 cannot be validated on a single GPU. It is an NCCL process group
   keyed by device ordinal, so it needs one real GPU per rank: Ray oversubscription
   (--num-gpus=4 on one card) gives "CUDA error: invalid device ordinal", and
   CUDA_VISIBLE_DEVICES=0,0,0,0 gives the same plus "Triton ... 0 active driver(s)
   found" (the workers see no device at all). In contrast our own engine's pipeline
   is process-level and device-agnostic (each process loads its own layers and
   exchanges activations over TCP), which is why four stages ran on a single GPU
   earlier. Functional validation of PP>1 therefore requires at least two real GPUs.

Added a single-machine validation-boundary table to the plan doc: verified here are
single-instance output, the native prefix cache, LMCache integration, startup cost,
and the tier-0/tier-1 drill; PP>1, cross-machine TOTP, WiFi measurements and
cross-machine NCCL require real hardware.
MSG
git -c safe.directory='*' push origin main 2>&1 | tail -2
echo "pushed $(git rev-parse --short HEAD)"

echo
echo "=== final cleanup: leave nothing running ==="
pkill -f gateway_drill 2>/dev/null || true
pkill -f vllm_gateway 2>/dev/null || true
pkill -f 'vllm serve' 2>/dev/null || true
"$HOME/venvs/vllm/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 4
pgrep -af 'vllm|gateway_drill|ray::' | head -5 || echo "  nothing running"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  gpu: /'
echo
df -h / /mnt/d | tail -2
git log --oneline -3
