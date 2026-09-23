#!/usr/bin/env bash
# Commit the tier-1 decision + WSL port fixes, push, then report install progress.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

git add -A
echo "=== staged ==="
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: tier-1 recovery confirmed (replay the concatenated prompt); fix WSL bridge notes

User decision: node departure is handled by tier 0 (bubble restart) PLUS tier 1 --
the in-flight request is recovered by appending what has already been generated to
the prompt and re-prefilling it. So in-flight requests are not dropped.

deploy/vllm-lmcache-4node-plan.md section 5 rewritten around that decision:
- the full 7-step flow (detect, stop admitting, drain, re-form on survivors,
  warm up, replay, resume admitting)
- the four things the entry gateway must do, because tier 0 + tier 1 live entirely
  in that gateway (accumulate the generated text, detect failure, stop-admit+drain,
  re-submit the concatenated prompt and forward only the NEW tokens so the client
  stream never breaks)
- a technical point that must be checked once vLLM is installed: replay by TEXT
  re-tokenises, and detokenise->tokenise is not guaranteed to be identity, so
  exact replay should use token ids (prompt_token_ids / a tokenize endpoint);
  otherwise replay by text and state the caveat in the experiment
- LMCache's role: if the prefix KV survives the restart, the replay prefill hits
  the cache and is nearly free; otherwise it is a full prefill (0.77 s for the
  Azure-conv profile, 4.8 s for Mooncake at 7B). The D phase of the smoke test
  therefore directly determines the size of the tier-1 bubble.
- the four quantities that set the bubble (weight load, drain, Ray re-form,
  replay prefill), and why tier 2 (mirroring) is not being built.

AGENTS.md (workspace bridge notes) corrected against measurements made today:
- the SSH port is 2222, not 22. /etc/ssh/sshd_config sets Port 2222 and systemd's
  sshd-socket-generator propagates it into ssh.socket, so the old note that
  "changing Port in sshd_config has no effect" is wrong on this systemd version.
- documents the WSL idle-shutdown failure mode and its exact symptom
  (Connection established followed by kex_exchange_identification: write:
  Connection refused because the Windows relay is up but sshd inside is not),
  that the agent cannot recover it itself, and the one command the user must run.
- records that Windows-side git push is impossible (git spawns sh.exe ->
  CreateFileMapping Win32 error 5) so pushes must go through WSL, while local
  add/commit on Windows works -- which is what let us keep committing while WSL
  was down.
- records the Chinese-in-script workaround (drop the script outside the repo and
  run `bash "$WS/.tmp/x.sh"`), the "always use a script file, never inline quotes"
  rule, and that pip caches must live on /mnt/d to keep the ext4.vhdx from growing.
EOF
git -c user.name="yy-chen0128" -c user.email="yy-chen0128@users.noreply.github.com" \
  commit -F "$MSGFILE" >/dev/null
rm -f "$MSGFILE"

echo
echo "=== push ==="
git -c safe.directory='*' push origin main 2>&1 | tail -3

echo
echo "=== vLLM install progress ==="
if [ -f .models/logs/vllm_install.pid ] && kill -0 "$(cat .models/logs/vllm_install.pid)" 2>/dev/null; then
  echo "pid $(cat .models/logs/vllm_install.pid): ALIVE"
else
  echo "install: finished or died"
fi
echo "venv size: $(du -sh "$HOME/venvs/vllm" 2>/dev/null | cut -f1)"
echo "--- log tail ---"
tail -c 400 .models/logs/vllm_install.log
echo
echo "--- disk ---"
df -h / /mnt/d | tail -3
