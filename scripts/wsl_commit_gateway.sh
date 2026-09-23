#!/usr/bin/env bash
# Commit the tier-0/tier-1 gateway and its tests. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

git add -A
echo "=== staged ==="
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
feat: entry gateway implementing tier 0 (bubble restart) + tier 1 (replay)

vLLM cannot express "stop admitting / drain / bring in-flight requests back after
the pipeline is rebuilt", so all of tier 0 + tier 1 lives in a gateway in front of
it. New: edge_llm_scheduler/gateway/vllm_gateway.py

What it does:
- proxies /v1/completions and /v1/chat/completions to vLLM, streaming
- accumulates what has already been generated for each request (this is also the
  progress reading that progress_tokens was missing)
- on any downstream failure: mark the pipeline unavailable (drain) and stop
  admitting new requests with 503
- wait for the pipeline to come back (POST /admin/resume, or the upstream
  /v1/models becoming reachable again)
- replay: rebuild the request body by appending the generated text to the prompt
  (for chat, append an assistant message) and re-submit, forwarding only the NEW
  text so the client's SSE stream never breaks
- admin surface: GET /admin/status, POST /admin/drain, POST /admin/resume

One extra thing that turns the tokenizer caveat into a metric: the replayed
output is compared against what was already sent, and a mismatch is counted as
replay_diverged in /admin/status. Replaying by TEXT re-tokenises and is not
guaranteed to be identity, so this makes replay fidelity a number per experiment
instead of a verbal risk (see docs/deploy/vllm-lmcache-4node-plan.md 5.3).

Tests: edge_llm_scheduler/tests/test_gateway_replay.py, 9 cases covering prompt
concatenation for both endpoints, list-prompts, stream_options stripping, the
longest-common-prefix skip (exact and diverging), delta/full extraction for both
endpoints, and the drain/resume state machine. Full suite: 107 passed, 3 skipped.

Also: the gateway's web deps (fastapi 0.141.1 / uvicorn 0.53.0 / httpx 0.28.1)
are now installed into ~/venvs/pair, and the plan doc records where each of the
four gateway responsibilities is implemented.

Note on the environment: the machine's pip config points at
http://mirrors.aliyun.com/pypi/simple/ with install-scoped trusted-host, which
current pip ignores, so the first vLLM install attempt stalled at 3 MB. The
install now uses https://pypi.tuna.tsinghua.edu.cn/simple explicitly and logs
unbuffered. A stale pip from the first attempt was still running against the same
venv and was killed (racing installs into one venv corrupt it).
EOF
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -F "$MSGFILE" >/dev/null
rm -f "$MSGFILE"

echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -3

echo
echo "=== vLLM install progress ==="
for pid in $(pgrep -f 'venvs/vllm/bin/pip' 2>/dev/null); do
  echo "  pip $pid alive"
done
echo "venv size: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)"
tail -c 300 .models/logs/vllm_install2.log
echo
df -h / /mnt/d | tail -3
