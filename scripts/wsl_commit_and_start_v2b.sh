#!/usr/bin/env bash
# Commit the local_disk negative result + the c10d warning, then start V2
# (PP=4 on one GPU) since V1 is done. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -F - <<'MSG'
docs: local_disk does NOT make the LMCache prefix cache survive a restart

Follow-up to the V1 measurement. With local_cpu (2 GB) plus local_disk configured:
  C2 LMCache fresh          cold 402.9 ms, warm 20.7 / 20.3 ms
  D2 LMCache after restart  cold 462.3 ms, warm 22.1 / 21.3 ms
and the decisive evidence: /mnt/d/lmcache_disk holds 0 bytes -- LMCache never wrote
to disk at all. The CPU tier is far larger than this request's KV (~37 MB for 3011
tokens on a 0.5B model), so no eviction ever happened and the disk tier stayed
empty. In other words local_disk is an eviction/overflow tier, not a persistence
tier. Corrected conclusion: with LMCache 0.3.5's local backends the prefix KV cannot
survive a process restart; only a remote backend (lmcache_server plus
remote_url="lm://host:port") can, which is exactly what the four-machine plan needs
and which can be validated on a single machine first.

Also recorded a warning that matters for the four machines:
  [c10d] The hostname of the client socket cannot be retrieved. err=-3
Mirrored networking leaves WSL unable to resolve its own hostname, and vLLM's
distributed init uses c10d/TCPStore. Harmless on one machine, but multi-node PP may
hang or pick the wrong address, so MASTER_ADDR / VLLM_HOST_IP must be set explicitly.
Added to the interconnect troubleshooting table.

Next: V2 (PP=4 on one GPU) and V2b (the same thing over a local Ray cluster, which
exercises the code path the four machines will use).
MSG
git -c safe.directory='*' push origin main 2>&1 | tail -2
echo "pushed $(git rev-parse --short HEAD)"

echo
echo "=== make sure no server is left running ==="
pkill -f 'vllm serve' 2>/dev/null && echo "  killed" || echo "  none running"
sleep 3

echo
echo "=== start V2b: local Ray cluster + PP=4 (the four-machine code path) ==="
cat > .models/logs/run_v2b_ray.sh <<'OUTER'
#!/usr/bin/env bash
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"; PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct; PORT=8020
OUT=.models/logs/v2b; mkdir -p "$OUT"
export PYTHONPATH=.

echo "### ray version"; "$VENV/bin/ray" --version
echo "### stop any old ray"
"$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 3

echo "### start head with 4 (virtual) GPUs so PP=4 can be placed on this one card"
"$VENV/bin/ray" start --head --num-gpus=4 --port=6379 \
   --min-worker-port=10002 --max-worker-port=10100 > "$OUT/ray_start.log" 2>&1
sleep 8
"$VENV/bin/ray" status 2>&1 | head -12

echo
echo "### vllm serve with pipeline-parallel-size=4 over ray"
export VLLM_HOST_IP=127.0.0.1
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
   --pipeline-parallel-size 4 --distributed-executor-backend ray \
   --max-model-len 4096 --gpu-memory-utilization 0.30 \
   > "$OUT/server.log" 2>&1 &
echo "server pid $!"

ready=0
for i in $(seq 1 72); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 5
done
if [ "$ready" = "1" ]; then
  echo "PP=4 (ray) READY after ~$((i*5))s"
  grep -iE 'pipeline|PP|rank|world_size|Starting vLLM|init engine' "$OUT/server.log" | head -15
  echo
  "$PY" scripts/vllm_bench_prefix.py --base-url "http://127.0.0.1:$PORT" \
      --prefix-tokens 500 --requests 2 --max-tokens 16 --label PP4-ray 2>&1 | tail -6
else
  echo "PP=4 (ray) NOT READY after $((i*5))s; last 30 lines:"
  tail -30 "$OUT/server.log"
fi
echo "### stop"
pkill -f 'vllm serve' 2>/dev/null; sleep 3
"$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
echo "### V2B_DONE"
OUTER
nohup bash .models/logs/run_v2b_ray.sh > .models/logs/v2b/driver.log 2>&1 &
echo $! > .models/logs/v2b.pid
echo "V2b started pid $(cat .models/logs/v2b.pid); log .models/logs/v2b/driver.log"
