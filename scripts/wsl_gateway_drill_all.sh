#!/usr/bin/env bash
# Run the tier-0 + tier-1 gateway drill end to end (PP=1, which is all it needs).
#   vLLM -> our gateway -> a streaming client, with vLLM killed and restarted mid-stream.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
VENV="$HOME/venvs/vllm"; PAIRPY="$HOME/venvs/pair/bin/python"
MODEL=.models/Qwen2.5-0.5B-Instruct
VPORT=8000; GPORT=8100
OUT=.models/logs/drill; mkdir -p "$OUT"

echo "=== cleanup ==="
pkill -f 'vllm serve' 2>/dev/null || true
pkill -f 'vllm_gateway' 2>/dev/null || true
sleep 3

echo "=== compile-check the client FIRST (a broken f-string cost us a run before) ==="
"$PAIRPY" -m py_compile scripts/gateway_drill.py && echo "  gateway_drill.py: OK" || exit 1
"$PAIRPY" -m py_compile scripts/vllm_bench_prefix.py && echo "  vllm_bench_prefix.py: OK" || exit 1

# helper the drill calls to kill+restart vLLM (simulating the pipeline rebuild)
cat > .models/logs/vllm_restart.sh <<EOF
#!/usr/bin/env bash
set -u
cd "$HOME/pair" || exit 1
pkill -f 'vllm serve' 2>/dev/null || true
sleep 4
nohup "$VENV/bin/vllm" serve "$MODEL" --port $VPORT \\
   --max-model-len 8192 --gpu-memory-utilization 0.55 \\
   > "$OUT/server_after_restart.log" 2>&1 &
for i in \$(seq 1 70); do
  curl -sf "http://127.0.0.1:$VPORT/v1/models" >/dev/null 2>&1 && { echo "restarted ready after ~\$((i*5))s"; exit 0; }
  sleep 5
done
echo "restart did NOT become ready"
exit 1
EOF
chmod +x .models/logs/vllm_restart.sh

echo "=== start vLLM (PP=1) ==="
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$VPORT" \
   --max-model-len 8192 --gpu-memory-utilization 0.55 > "$OUT/server.log" 2>&1 &
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$VPORT/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://127.0.0.1:$VPORT/v1/models" >/dev/null || { echo "vLLM not ready"; tail -10 "$OUT/server.log"; exit 1; }
echo "  vLLM ready (~$((i*5))s)"

echo "=== start the gateway (stall timeout 10s so recovery fires quickly) ==="
PYTHONPATH=. nohup "$PAIRPY" -m edge_llm_scheduler.gateway.vllm_gateway \
   --upstream "http://127.0.0.1:$VPORT" --port "$GPORT" --stall-timeout 10 \
   > "$OUT/gateway.log" 2>&1 &
sleep 8
curl -s "http://127.0.0.1:$GPORT/admin/status" | head -c 300; echo

echo
echo "=== run the drill: kill vLLM 3s into a 400-token stream ==="
"$PAIRPY" scripts/gateway_drill.py --gateway-url "http://127.0.0.1:$GPORT" \
    --model "$MODEL" --max-tokens 400 --kill-after 3.0 2>&1 | tail -22

echo
echo "=== gateway log (recovery decisions) ==="
grep -iE 'failed|replay|down|resume|forwarding|timeout' "$OUT/gateway.log" | tail -12

echo
echo "=== gateway status after the drill ==="
curl -s "http://127.0.0.1:$GPORT/admin/status"; echo

echo
echo "=== cleanup ==="
pkill -f 'vllm_gateway' 2>/dev/null || true
pkill -f 'vllm serve' 2>/dev/null || true
echo "### DRILL_DONE"
