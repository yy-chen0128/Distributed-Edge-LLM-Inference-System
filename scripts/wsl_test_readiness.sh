#!/usr/bin/env bash
# Test the readiness self-check on THIS machine, then test the fleet planner with
# synthetic reports that exercise the heterogeneity rules.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
PY="$HOME/venvs/pair/bin/python"
OUT=.models/logs/readiness; mkdir -p "$OUT"

echo "=== compile checks (a broken f-string has cost us runs before) ==="
"$PY" -m py_compile scripts/check_node_readiness.py \
                  scripts/fleet_plan_from_readiness.py && echo "  OK" || exit 1

echo
echo "############ 1. self-check on THIS machine ############"
"$PY" scripts/check_node_readiness.py --json-out "$OUT/node_readiness_$(hostname).json" 2>&1 | tail -45
echo "exit=$?"

echo
echo "############ 2. fleet planner on 4 synthetic reports ############"
# Build four plausible laptops: this one (8GB / CUDA 12.7), a 6GB at CUDA 12.6,
# two 4GB at CUDA 12.4 -- i.e. heterogeneous VRAM AND driver.
"$PY" - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
def mk(host, gpu, vram, drv, cuda, cap, stack, mirrored=True, ping="0"):
    usable = round(max(0.0, vram - 0.8), 2)
    return {
        "hostname": host,
        "facts": {
            "hostname": host, "os": "Ubuntu 24.04", "is_wsl": True,
            "wsl_mirrored": mirrored, "gpu": gpu, "vram_gb": vram,
            "driver_version": drv, "cuda_version": cuda, "compute_cap": cap,
            "max_stack": stack, "vram_usable_gb": usable,
            "7b_fp16_layers_fit": int(usable / 0.4661),
            "7b_int4_layers_fit": int(usable / 0.1280),
            "model_recommendation": "?",
            "ports_busy": [], "local_models": {},
            "peer_ping_loss_pct": ping, "peer_open_ports": [2222],
        },
        "verdicts": {"wsl-networking": "PASS" if mirrored else "FAIL",
                     "stack-ceiling": "PASS" if cuda >= 12.4 else "FAIL",
                     "gpu": "PASS", "peer-ping": "PASS" if ping == "0" else "FAIL"},
    }
cases = [
    mk("laptop-A", "RTX 4060 Laptop", 8.0, "566.24", 12.7, "8.9", "vllm 0.10.1 / 0.9.2 (torch 2.7.x/cu126)"),
    mk("laptop-B", "RTX 3060 Laptop", 6.0, "560.94", 12.6, "8.6", "vllm 0.10.1 / 0.9.2 (torch 2.7.x/cu126)"),
    mk("laptop-C", "RTX 3050 Laptop", 4.0, "552.44", 12.4, "8.6", "vllm 0.8.5 (torch 2.6/cu124)"),
    mk("laptop-D", "GTX 1650", 4.0, "552.44", 12.4, "7.5", "vllm 0.8.5 (torch 2.6/cu124)", mirrored=False, ping="100"),
]
for c in cases:
    with open(os.path.join(out, f"synthetic_{c['hostname']}.json"), "w") as fh:
        json.dump(c, fh, indent=2)
print(f"  wrote {len(cases)} synthetic reports to {out}/")
EOF
echo
"$PY" scripts/fleet_plan_from_readiness.py "$OUT"/synthetic_*.json 2>&1 | tail -55
