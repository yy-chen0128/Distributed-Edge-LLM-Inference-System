#!/usr/bin/env bash
# Commit the elasticity cost/benefit analysis. ASCII ONLY.
set -eu
cd "$HOME/pair"

git add -A
git diff --cached --stat

MSGFILE=$(mktemp)
cat > "$MSGFILE" <<'EOF'
docs: elasticity cost/benefit -- R1 vs R2, and the cheapest recovery

New doc docs/design/elasticity-cost-benefit.md. It separates two gaps that are
easy to conflate:
- R1 = changing the layer split at runtime (every engine fixes it at load/config time)
- R2 = not losing in-flight requests (independent of whether the split can change)
R1's cost is weight loading (measured 35.7 s for 7B/4 layers, ~126 s for a 12-layer
stage); R2's cost is state rebuild (1-5 s). So R1 dominates.

Four options when a node leaves (drop / replay / move KV / mirror), costed per
request using the measured per-layer constants (7B: 72 ms per generated token,
0.61 ms per prompt token per full model) against four real workload profiles:
  Azure code (1500/13):        1.85 s drop vs 0.99 s replay
  Azure conv (1020/129):       9.91 s vs 0.77 s   (12.9x)
  Mooncake    (7590/182):     17.70 s vs 4.79 s
  Codex       (68329/520):    79.0 s  vs 41.9 s
Key points:
- The cheapest correct recovery is to append the generated tokens to the prompt
  and replay one prefill. It needs no KV export/import, no cooperation from the
  departed machine, and no redundancy -- it reuses the existing prefill path.
  It only requires the control plane to record generated tokens, which is the
  same bookkeeping that progress_tokens needs.
- Moving KV is cheap because only the departing stage's layers move:
  2048 B/layer/token x 4 layers x (P+g) = 12-564 MB, i.e. 7x less than the whole
  model's KV for that position.
- A node departure interrupts K in-flight requests, not one: the pipeline breaks
  and every request stalls at its current token.
- Replay's saving is exactly the decode segment, so it pays most for long outputs.

Benefit assessment: benefit = waste avoided x K x departure rate, and the
departure rate has NO public data (churn is the known gap), so it must be an
experimental variable and plotted as a curve. Under a throughput-only metric the
benefit at low churn is under 1%; under a service-quality metric (no dropped
requests, no whole-request p99 spikes) it is worth much more. Recommendation:
implement tier 0 (stop-admit, drain, re-form, restart = vLLM semantics) plus
tier 1 (replay, about half a day of wiring), skip mirroring (tier 2).

Also records the answer on reusing vLLM: its functional semantics equal tier 0
(restart, lose in-flight), but it requires identical execution environments across
nodes ("to hide host heterogeneity"), fixes the split at configuration time, and
has no drain/blue-green -- so we reuse its ideas and use it as a datacenter
baseline rather than as our engine.

Priority correction: the thing to optimise first is the 35-126 s weight load, not
recovery, because the load is paid on every topology change while replay costs 1-5 s.
EOF
git commit -F "$MSGFILE"
rm -f "$MSGFILE"

echo
git log --oneline -3
echo
echo "=== push ==="
git push origin main
git status -sb | head -2
git ls-files | wc -l
