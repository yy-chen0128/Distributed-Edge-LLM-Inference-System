"""Node readiness self-check for joining the vLLM + LMCache inference cluster.

Run this INSIDE WSL (or native Ubuntu) on every candidate machine:

    python scripts/check_node_readiness.py --peer <head-ip> \
        --json-out node_readiness_<name>.json

It answers two different questions, and keeps them separate on purpose:

  A. "Can this machine satisfy the shared software stack?"   -> driver/CUDA version
     (the fleet is capped by the OLDEST machine, because one wheel set is installed
     everywhere);
  B. "What can this machine contribute?"                      -> VRAM, compute cap,
     link class, and how many layers / what context it can hold.

Every check reports the raw value, a verdict (PASS / WARN / FAIL / INFO) and, when
relevant, what it implies for the fleet. Nothing here mutates the machine.

ASCII-only output on purpose: this may be run over ssh and pasted back.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from typing import Any, Optional

# ---------------------------------------------------------------- stack rules
# Wheel CUDA version -> the driver's reported CUDA version it needs. This is the
# table from docs/deploy/four-machine-interconnect.md 1.4; keep them in sync.
#
#   vllm 0.8.5  -> torch 2.6.0 (cu124)  -> driver CUDA >= 12.4
#   vllm 0.9.2  -> torch 2.7.0 (cu126)  -> driver CUDA >= 12.6
#   vllm 0.10.1 -> torch 2.7.1 (cu126)  -> driver CUDA >= 12.6   <-- current choice
#   vllm 0.10.2 -> torch 2.8.0 (cu128)  -> driver CUDA >= 12.8
#   vllm 0.30   -> torch 2.13+ (cu130)  -> driver CUDA >= 13.0
STACK_CANDIDATES = [
    (13.0, "vllm>=0.11 (torch 2.13+/cu130)"),
    (12.8, "vllm 0.10.2 (torch 2.8/cu128)"),
    (12.6, "vllm 0.10.1 / 0.9.2 (torch 2.7.x/cu126)"),
    (12.4, "vllm 0.8.5 (torch 2.6/cu124)"),
]

# VRAM needed per stage for 7B, measured from the real safetensors headers:
#   466.1 MB/layer, plus the vocabulary matrix (1.09 GB) on the first stage and the
#   output head (1.09 GB) on the last stage. Reserve 0.8 GB for activations/KV/frag.
PER_LAYER_GB_7B_FP16 = 0.4661
PER_LAYER_GB_7B_INT4 = 0.1280
VOCAB_GB_7B = 1.09


def run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


def which_nvidia_smi() -> Optional[str]:
    for cand in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        path = shutil.which(cand) if "/" not in cand else (cand if os.path.exists(cand) else None)
        if path:
            return path
    return None


# ------------------------------------------------------------------- checks


def check_system() -> dict:
    out: dict[str, Any] = {"checks": [], "facts": {}}
    kernel = platform.release()
    is_wsl = "microsoft" in kernel.lower() or "wsl" in kernel.lower()
    distro = "?"
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    distro = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    out["facts"].update({"os": distro, "kernel": kernel, "is_wsl": is_wsl,
                         "python": platform.python_version(),
                         "hostname": socket.gethostname()})
    out["checks"].append(("system", "INFO", f"{distro} | kernel {kernel} | python "
                          f"{platform.python_version()} | wsl={is_wsl}"))

    # WSL networking: mirrored is required for other machines to reach services here
    if is_wsl:
        cfg = ""
        for path in glob.glob("/mnt/c/Users/*/.wslconfig"):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    cfg += fh.read()
            except OSError:
                pass
        mirrored = "networkingmode=mirrored" in cfg.replace(" ", "").lower()
        out["facts"]["wsl_mirrored"] = mirrored
        if mirrored:
            out["checks"].append(("wsl-networking", "PASS",
                                  "networkingMode=mirrored (services are reachable "
                                  "from other machines)"))
        else:
            out["checks"].append(("wsl-networking", "FAIL",
                                  "no .wslconfig with networkingMode=mirrored -> other "
                                  "machines CANNOT reach this machine's WSL; Ray/vLLM PP "
                                  "will not work. Add [wsl2] networkingMode=mirrored and "
                                  "run `wsl --shutdown`"))
    return out


def check_disk(need_gb: float = 20.0) -> dict:
    out: dict[str, Any] = {"checks": [], "facts": {}}
    for label, path in (("root(venv)", "/"), ("models(mnt_d)", "/mnt/d")):
        try:
            st = os.statvfs(path if os.path.exists(path) else "/")
            free_gb = st.f_bavail * st.f_frsize / 1e9
        except OSError:
            continue
        out["facts"][f"free_gb_{label}"] = round(free_gb, 1)
        verdict = "PASS" if free_gb >= need_gb else ("WARN" if free_gb >= 12 else "FAIL")
        out["checks"].append((f"disk-{label}", verdict, f"{free_gb:.1f} GB free "
                              f"(need >= {need_gb:.0f} GB for venv ~9 GB + models)"))
    return out


def check_gpu() -> dict:
    out: dict[str, Any] = {"checks": [], "facts": {}}
    smi = which_nvidia_smi()
    if not smi:
        out["checks"].append(("gpu", "FAIL", "nvidia-smi not found (no NVIDIA GPU or "
                              "driver not visible in WSL)"))
        return out
    rc, text = run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader,nounits"])
    if rc != 0 or not text.strip():
        out["checks"].append(("gpu", "FAIL", f"nvidia-smi failed: {text[:120]}"))
        return out
    first = text.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else "?"
    vram_mib = float(parts[1]) if len(parts) > 1 and parts[1].replace(".", "").isdigit() else 0.0
    driver = parts[2] if len(parts) > 2 else "?"
    cap = parts[3] if len(parts) > 3 else "?"
    vram_gb = vram_mib / 1024.0

    # the driver's CUDA version is in the header line: "CUDA Version: 12.7"
    rc2, hdr = run([smi])
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", hdr)
    cuda = float(m.group(1)) if m else 0.0

    out["facts"].update({"gpu": name, "vram_gb": round(vram_gb, 1),
                         "driver_version": driver, "cuda_version": cuda,
                         "compute_cap": cap})
    out["checks"].append(("gpu", "INFO", f"{name} | {vram_gb:.1f} GB | driver {driver} "
                          f"| CUDA {cuda} | sm{cap}"))

    # A. fleet-level stack ceiling
    best = None
    for need, label in STACK_CANDIDATES:
        if cuda >= need:
            best = (need, label)
            break
    if best:
        out["facts"]["max_stack"] = best[1]
        out["checks"].append(("stack-ceiling", "PASS",
                              f"CUDA {cuda} supports up to: {best[1]}"))
    else:
        out["facts"]["max_stack"] = None
        out["checks"].append(("stack-ceiling", "FAIL",
                              f"CUDA {cuda} is below 12.4 -> cannot run any current "
                              f"vLLM/torch wheel; upgrade the Windows-side NVIDIA driver"))

    # B. what this machine can hold (heterogeneity facts)
    reserve = 0.8
    avail = max(0.0, vram_gb - reserve)
    out["facts"]["vram_usable_gb"] = round(avail, 2)
    fp16_layers = int(avail / PER_LAYER_GB_7B_FP16)
    int4_layers = int(avail / PER_LAYER_GB_7B_INT4)
    out["facts"]["7b_fp16_layers_fit"] = fp16_layers
    out["facts"]["7b_int4_layers_fit"] = int4_layers
    if fp16_layers >= 7:
        rec = "7B fp16 (>=7 layers/stage)"
    elif int4_layers >= 7:
        rec = "7B AWQ/GPTQ int4 (fp16 would OOM: needs 7 layers)"
    elif int4_layers >= 4:
        rec = "7B int4 with fewer layers than 7 (needs a capacity-weighted split)"
    else:
        rec = "0.5B/1.5B/3B class only"
    out["facts"]["model_recommendation"] = rec
    out["checks"].append(("model-tier", "INFO",
                          f"usable {avail:.2f} GB -> 7B fp16 fits {fp16_layers} layers/stage, "
                          f"int4 fits {int4_layers} -> recommended: {rec}"))
    # warning: vocabulary matrices land on the first and last stage
    if fp16_layers >= 7:
        fixed = VOCAB_GB_7B
        out["checks"].append(("vocab-cost", "INFO",
                              f"first/last stage additionally carry ~{fixed:.2f} GB of "
                              f"vocabulary/output-head each (tie_word_embeddings=false "
                              f"for 7B)"))

    # compute capability implications
    try:
        capf = float(cap)
    except ValueError:
        capf = 0.0
    notes = []
    if capf >= 8.9:
        notes.append("FP8 kernels available")
    else:
        notes.append("no FP8 (needs sm89+)")
    if capf >= 8.0:
        notes.append("FlashAttention-2 era kernels available")
    if capf < 7.5:
        notes.append("very old: many vLLM kernels unavailable")
    out["checks"].append(("compute-cap", "INFO", f"sm{cap}: " + "; ".join(notes)))
    return out


def check_torch_stack(venv: str) -> dict:
    """Import torch/vllm/ray/lmcache in the given venv and report versions."""
    out: dict[str, Any] = {"checks": [], "facts": {}}
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        out["checks"].append((f"venv:{venv}", "WARN", "not present (not installed yet)"))
        return out
    code = (
        "import json\n"
        "d={}\n"
        "try:\n"
        "    import torch; d['torch']=torch.__version__; d['torch_cuda']=torch.version.cuda;"
        " d['cuda_available']=torch.cuda.is_available()\n"
        "except Exception as e: d['torch']='FAILED:'+type(e).__name__\n"
        "for m in ('vllm','ray','lmcache','transformers','tokenizers'):\n"
        "    try:\n"
        "        mod=__import__(m); d[m]=getattr(mod,'__version__','?')\n"
        "    except Exception as e: d[m]='FAILED:'+type(e).__name__\n"
        "try:\n"
        "    if d.get('lmcache') in ('?', None):\n"
        "        from lmcache._version import __version__ as lv; d['lmcache']=lv\n"
        "except Exception: pass\n"
        "print(json.dumps(d))\n"
    )
    rc, text = run([py, "-c", code], timeout=180)
    if rc != 0:
        out["checks"].append((f"venv:{venv}", "WARN", f"import probe failed: {text[:120]}"))
        return out
    try:
        d = json.loads(text.strip().splitlines()[-1])
    except (ValueError, IndexError):
        out["checks"].append((f"venv:{venv}", "WARN", f"unparsable: {text[:120]}"))
        return out
    out["facts"][os.path.basename(venv)] = d
    ok = d.get("cuda_available") is True
    out["checks"].append((f"venv:{venv}", "PASS" if ok else "FAIL",
                          f"torch={d.get('torch')} cuda_build={d.get('torch_cuda')} "
                          f"cuda_available={d.get('cuda_available')} vllm={d.get('vllm')} "
                          f"ray={d.get('ray')} lmcache={d.get('lmcache')} "
                          f"transformers={d.get('transformers')}"))
    # The transformers<5 pin only applies to the venv that runs vLLM. Our own
    # engine's venv legitimately uses transformers 5.x, so do not flag it there.
    tf = str(d.get("transformers", ""))
    is_vllm_venv = "vllm" in os.path.basename(venv).lower() and d.get("vllm") not in (
        None, "?", "")
    if is_vllm_venv and tf and not tf.startswith("FAILED"):
        try:
            major = int(tf.split(".")[0])
            if major >= 5:
                out["checks"].append(("transformers-pin", "FAIL",
                                      f"transformers {tf} >= 5 breaks vLLM 0.10.1 "
                                      f"(all_special_tokens_extended); pin <5"))
            else:
                out["checks"].append(("transformers-pin", "PASS",
                                      f"transformers {tf} (<5, correct for vLLM 0.10.1)"))
        except ValueError:
            pass
    return out


def check_ports(ports: list[int]) -> dict:
    out: dict[str, Any] = {"checks": [], "facts": {}}
    busy = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            free = True
        except OSError:
            free = False
        finally:
            sock.close()
        if not free:
            busy.append(port)
    out["facts"]["ports_busy"] = busy
    if busy:
        out["checks"].append(("ports", "WARN", f"in use on 0.0.0.0: {busy} "
                              f"(needed by ray/vllm/gateway = 6379/8000/8100)"))
    else:
        out["checks"].append(("ports", "PASS",
                              f"all of {ports} bindable on 0.0.0.0"))
    return out


def check_pip_config() -> dict:
    out: dict[str, Any] = {"checks": [], "facts": {}}
    cfg = ""
    for path in ("/etc/pip.conf", os.path.expanduser("~/.config/pip/pip.conf"),
                 os.path.expanduser("~/.pip/pip.conf")):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    cfg += f"[{path}]\n" + fh.read()
            except OSError:
                pass
    out["facts"]["pip_config"] = cfg.strip()[:400]
    if "index-url" in cfg and "http://" in cfg and "https://" not in cfg:
        out["checks"].append(("pip-config", "WARN",
                              "pip.conf points at an http mirror; current pip ignores "
                              "install-scoped trusted-host and can stall -> always pass "
                              "-i https://pypi.tuna.tsinghua.edu.cn/simple explicitly"))
    else:
        out["checks"].append(("pip-config", "PASS" if cfg else "INFO",
                              "no http-mirror trap detected" if cfg else "no pip.conf"))
    return out


def check_peer(peer: str, ports: list[int]) -> dict:
    out: dict[str, Any] = {"checks": [], "facts": {}}
    rc, text = run(["ping", "-c", "3", "-W", "2", peer], timeout=20)
    loss = "?"
    m = re.search(r"(\d+)% packet loss", text)
    if m:
        loss = m.group(1)
    out["facts"]["peer_ping_loss_pct"] = loss
    out["checks"].append(("peer-ping", "PASS" if loss == "0" else "FAIL",
                          f"ping {peer}: {loss}% loss (campus WiFi often isolates "
                          f"clients -> the cluster cannot form)"))
    reachable = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            sock.connect((peer, port))
            reachable.append(port)
        except OSError:
            pass
        finally:
            sock.close()
    out["facts"]["peer_open_ports"] = reachable
    out["checks"].append(("peer-ports", "INFO",
                          f"TCP connect to {peer}: open {reachable or 'none'} "
                          f"of {ports}"))
    return out


def check_models(models_dir: str) -> dict:
    """For any local model, report per-layer bytes and KV/token (the numbers the
    layer-split decision needs)."""
    out: dict[str, Any] = {"checks": [], "facts": {}}
    found = {}
    for cfg_path in glob.glob(os.path.join(models_dir, "*", "config.json")):
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            continue
        name = os.path.basename(os.path.dirname(cfg_path))
        hidden = int(cfg.get("hidden_size") or 0)
        layers = int(cfg.get("num_hidden_layers") or 0)
        heads = int(cfg.get("num_attention_heads") or 0)
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        head_dim = int(cfg.get("head_dim") or (hidden // heads if heads else 0))
        inter = int(cfg.get("intermediate_size") or 0)
        vocab = int(cfg.get("vocab_size") or 0)
        tie = bool(cfg.get("tie_word_embeddings", False))
        if not (hidden and layers):
            continue
        per_layer_bytes = (hidden * hidden + 2 * hidden * (kv_heads * head_dim)
                           + hidden * hidden + 3 * hidden * inter) * 2
        kv_per_token = 2 * kv_heads * head_dim * 2 * layers
        found[name] = {
            "layers": layers,
            "per_layer_gb_fp16": round(per_layer_bytes / 1e9, 4),
            "vocab_gb_fp16": round(vocab * hidden * 2 / 1e9, 3),
            "tie_word_embeddings": tie,
            "kv_kb_per_token": round(kv_per_token / 1024.0, 2),
        }
    out["facts"]["local_models"] = found
    for name, info in found.items():
        out["checks"].append(("model-local", "INFO",
                              f"{name}: {info['layers']} layers, "
                              f"{info['per_layer_gb_fp16']} GB/layer fp16, vocab "
                              f"{info['vocab_gb_fp16']} GB, KV {info['kv_kb_per_token']} KB/token"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", default="", help="head node IP to test reachability against")
    ap.add_argument("--models-dir", default=".models")
    ap.add_argument("--venvs", default="vllm,pair",
                    help="comma-separated venv names under ~/venvs to probe")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    results: dict[str, Any] = {"hostname": socket.gethostname(),
                               "checks": [], "facts": {}}
    sections = [check_system(), check_gpu(), check_disk(), check_pip_config(),
                check_ports([6379, 8000, 8100, 10002, 10050])]
    for venv in [v.strip() for v in args.venvs.split(",") if v.strip()]:
        sections.append(check_torch_stack(os.path.expanduser(f"~/venvs/{venv}")))
    sections.append(check_models(args.models_dir))
    if args.peer:
        sections.append(check_peer(args.peer, [6379, 8000, 8100, 22, 2222]))

    for sec in sections:
        results["checks"].extend(sec["checks"])
        results["facts"].update(sec["facts"])

    print("=" * 78)
    print(f"node readiness self-check  host={results['hostname']}")
    print("=" * 78)
    icon = {"PASS": "[ok]  ", "WARN": "[warn]", "FAIL": "[FAIL]", "INFO": "[info]"}
    for name, verdict, detail in results["checks"]:
        print(f"{icon.get(verdict, '[?]   ')} {name:22s} {detail}")

    fails = [c for c in results["checks"] if c[1] == "FAIL"]
    warns = [c for c in results["checks"] if c[1] == "WARN"]
    print("-" * 78)
    print(f"FAIL={len(fails)}  WARN={len(warns)}")
    print()
    print("### SEND THESE BACK TO THE CONTROLLER ###")
    keys = ["hostname", "os", "is_wsl", "wsl_mirrored", "gpu", "vram_gb",
            "driver_version", "cuda_version", "compute_cap", "max_stack",
            "vram_usable_gb", "7b_fp16_layers_fit", "7b_int4_layers_fit",
            "model_recommendation", "ports_busy", "local_models",
            "peer_ping_loss_pct", "peer_open_ports"]
    for key in keys:
        if key in results["facts"]:
            print(f"  {key}: {results['facts'][key]}")

    if args.json_out:
        payload = {"hostname": results["hostname"], "facts": results["facts"],
                   "vestatus": results["checks"],
                   "verdicts": {name: v for name, v, _ in results["checks"]}}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nwritten: {args.json_out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
