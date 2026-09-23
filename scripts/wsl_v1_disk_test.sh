#!/usr/bin/env bash
# Decisive follow-up: does LMCache with local_disk survive a restart?
# C' = LMCache(local_cpu + local_disk) fresh; D' = same after a restart.
# If D' is warm, tier-1 replay gets much cheaper.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
OUT=.models/logs/v1_disk
mkdir -p "$OUT" /mnt/d/lmcache_disk

cat > "$OUT/lmcache.yaml" <<'YAML'
chunk_size: 256
local_cpu: true
max_local_cpu_size: 2
local_disk: /mnt/d/lmcache_disk
max_local_disk_size: 8
YAML

cat > .models/logs/run_v1_disk.sh <<'OUTER'
#!/usr/bin/env bash
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"; PY="$VENV/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct; PORT=8000; OUT=.models/logs/v1_disk
BASE="http://127.0.0.1:$PORT"
KVCFG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
export LMCACHE_CONFIG_FILE="$HOME/pair/.models/logs/v1_disk/lmcache.yaml"

start() {
  local tag="$1"; shift
  nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
     --max-model-len 8192 --gpu-memory-utilization 0.55 "$@" \
     > "$OUT/server_$tag.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && { echo "  [$tag] ready"; return 0; }
    sleep 5
  done
  echo "  [$tag] NOT READY"; tail -12 "$OUT/server_$tag.log"; return 1
}
stop() { kill "$SERVER_PID" 2>/dev/null; sleep 6; pkill -f 'vllm serve' 2>/dev/null; sleep 4; }
bench() {
  "$PY" scripts/vllm_bench_prefix.py --base-url "$BASE" \
     --prefix-tokens 3000 --requests 3 --max-tokens 16 --label "$1" 2>&1 \
     | tee "$OUT/bench_$1.log"
}

echo "=== disk cache dir before ==="
du -sh /mnt/d/lmcache_disk 2>/dev/null || echo "  (empty)"
ls -la /mnt/d/lmcache_disk 2>/dev/null | head -5

echo
echo "########## C2: LMCache(local_cpu + local_disk), fresh ##########"
start C2 && bench C2-lmcache-disk-warm
echo
echo "=== disk cache dir after C2 ==="
du -sh /mnt/d/lmcache_disk 2>/dev/null
stop

echo
echo "########## D2: restart -> does the DISK cache make it warm? ##########"
start D2 && bench D2-lmcache-disk-restart
echo
echo "=== disk cache dir after D2 ==="
du -sh /mnt/d/lmcache_disk 2>/dev/null
stop

echo
echo "=== SUMMARY (compare with the local_cpu-only run) ==="
for f in "$OUT"/bench_*.log; do
  [ -f "$f" ] || continue
  printf '%-32s %s\n' "$(basename "$f" .log)" "$(grep -h 'JSON' "$f" | tail -1)"
done
echo "  (for reference, local_cpu-only was: C cold 560.1 / warm 27.9 ; D cold 482.0 / warm 27.0)"
echo "### V1_DISK_DONE"
OUTER

nohup bash .models/logs/run_v1_disk.sh > "$OUT/driver.log" 2>&1 &
echo $! > .models/logs/v1_disk.pid
echo "started pid $(cat .models/logs/v1_disk.pid); log: $OUT/driver.log"

echo
echo "=== meanwhile: commit the client fix + V1 findings ==="
git add -A
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -q -F - <<'MSG'
fix+docs: client f-string bug; V1 measurements and a corrected LMCache assumption

The bench client had a broken multi-line f-string, so all four phases of the smoke
test ran their servers but never issued a request (SyntaxError). Fixed, and the
client now compile-checks before a run. It also builds its prefix from " the"
(about one token per word) and reports the real prompt_tokens from the response
usage -- the earlier ctx00000-style prefix tokenised into several tokens each, so
the "3000-token" prompt exceeded max-model-len and the server correctly returned
HTTP 400.

V1 measurements (3011-token shared prefix, 3 requests per phase, 0.5B on an 8 GB
card), recorded in the plan doc:
  A plain vLLM fresh        cold 178.3 ms, warm 22.2 / 20.8 ms   (8.6x)
  B plain vLLM restarted    cold 165.6 ms, warm 19.5 / 23.0 ms   (cache died)
  C vLLM + LMCache fresh    cold 560.1 ms, warm 27.9 / 24.6 ms
  D vLLM + LMCache restarted cold 482.0 ms, warm 27.0 / 34.2 ms
Conclusions: vLLM's built-in prefix cache already gives 8.6x TTFT for a repeated
prefix with zero setup; the in-process cache dies on restart (the inherent cost of
tier-0 bubble restarts); and my earlier assumption that LMCache survives a restart
was WRONG -- with local_cpu only it is per-process CPU memory, and this config had
local_disk disabled, so D is cold again. Persistence needs local_disk, and
cross-instance sharing needs a remote lmcache_server. On a single instance with
local_cpu only, LMCache is a net negative (cold path +380 ms, warm path no better).

A follow-up run with local_disk enabled is already going, to test whether the disk
cache survives the restart.
MSG
git -c safe.directory='*' push origin main 2>&1 | tail -2
echo "pushed: $(git rev-parse --short HEAD)"
