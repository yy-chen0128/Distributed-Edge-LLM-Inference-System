#!/usr/bin/env bash
# Verify the two-machine scripts parse, commit LOCALLY (no push while the privacy
# work is paused), and report what is needed from the peer.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
PY="$HOME/venvs/pair/bin/python"

echo "=== compile checks ==="
"$PY" -m py_compile scripts/vllm_bench_prefix.py && echo "  vllm_bench_prefix.py OK"
for s in scripts/two_node_worker.sh scripts/two_node_head.sh scripts/two_node_this_machine_info.sh; do
  bash -n "$s" && echo "  $(basename "$s") syntax OK"
done

echo
echo "=== quick bench-client self-test (token/tpot columns) against no server ==="
"$PY" scripts/vllm_bench_prefix.py --base-url http://127.0.0.1:59999 --requests 1 \
    --prefix-tokens 5 --max-tokens 2 --label selfcheck 2>&1 | tail -3

echo
echo "=== commit locally (NOT pushing: privacy work is paused) ==="
git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -F - <<'MSG'
feat: two-machine bring-up -- scripts and runbook, plus TPOT in the bench client

Two machines means two real GPUs, which is what vLLM's PP>1 needs (its pipeline is an
NCCL process group keyed by device ordinal, so a single card cannot host two ranks).
So the two-machine setup is what finally validates PP=2 and, more importantly,
measures the per-token cost of crossing machines.

scripts/two_node_worker.sh (run on the worker):
- derives the interface and source address from the route to the head, and exports
  NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME / VLLM_HOST_IP from them. Both were real
  bugs before: WSL exposes several interfaces so NCCL picks the wrong one, and a
  wrong VLLM_HOST_IP (e.g. 127.0.0.1) makes vLLM's Ray placement group unsatisfiable
- checks ping, the head's Ray port, and that the model directory exists locally
  (each PP rank loads its own shard from its own disk)
- joins the cluster with ray start --address

scripts/two_node_head.sh (run on the head):
- preflight, then PHASE 1: PP=1 on this machine alone (baseline), then PHASE 2: Ray
  head + worker join + vllm serve --pipeline-parallel-size 2 over Ray
- waits for two nodes to appear in ray status before starting, so PP=2 is really
  cross-machine and not silently local
- optionally starts the worker over ssh (WORKER_SSH), otherwise tells you what to run
- prints a comparison table (warm TPOT, tok/s, warm TTFT) and the derived
  per-token cost of crossing machines, with the interpretation rule: if that delta
  dominates the per-token compute time, one-token-per-hop over this network is not
  worth it and the design must change

scripts/vllm_bench_prefix.py: now reports prompt/completion tokens, tok/s and TPOT
(time per output token after the first), and supports --ignore-eos so runs are
comparable regardless of where the model would have stopped.

docs/deploy/two-machine-bringup.md: the runbook. What to collect from the peer, which
model fits in two 8GB machines (0.5B for bring-up; 7B AWQ int4 as the real target;
7B fp16 does NOT fit -- 14 layers = 6.5 GB plus 1.09 GB of vocabulary on each end
exceeds 7.2 GB usable, so it needs a third machine or int4), the four hard
prerequisites, the link measurement, the startup order, the two-machine failure
drills (losing the worker means restarting as PP=1 on the head), and what to send back.

No personal identifiers in any of the new content: IPs and paths are placeholders.
MSG
echo
git log --oneline -2
echo
echo "=== unpushed (deliberately held) ==="
git log --oneline origin/main..main
