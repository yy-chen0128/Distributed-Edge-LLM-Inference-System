"""每台机器上跑一次：盘点硬件/软件，输出可直接填进集群配置的环境清单。

这是路线图"阶段 A：旧卡环境探测"的落地脚本。它回答四件事：
1. 这台机器**能装多大的模型分片**（显存、内存、磁盘）；
2. 这台机器的 GPU **支持哪些算子/量化路径**（按 vLLM 的架构门槛判定）；
3. 集群配置里该填什么（本机在局域网上的 IP、agent 端口建议）；
4. 这台机器适合当什么角色（强计算段 / 弱段 / KV 缓存 / 控制端）。

用法：
    python scripts/collect_env.py                      # 人类可读 + 写 JSON
    python scripts/collect_env.py --json-out env_alpha.json
    python scripts/collect_env.py --node-id alpha --port 9100
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def run(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"<失败: {type(exc).__name__}>"


# ---------------------------------------------------------------- OS / 环境

def probe_os() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "in_wsl": "microsoft" in platform.release().lower()
        or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"),
    }
    if info["system"] == "Windows":
        info["windows_build"] = platform.version()
        info["os_caption"] = run(["cmd", "/c", "ver"])
        # WSL 是否可用（不进入发行版，只问版本）
        info["wsl"] = run(["wsl.exe", "--version"]).replace("\x00", "")[:200] or "<未安装>"
    return info


def probe_cpu_mem() -> dict:
    info = {"cpu_cores": os.cpu_count()}
    try:
        import psutil  # 可选

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
        return info
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                info["ram_total_gb"] = round(stat.ullTotalPhys / 1e9, 1)
                info["ram_available_gb"] = round(stat.ullAvailPhys / 1e9, 1)
        except Exception as exc:  # noqa: BLE001
            info["ram_error"] = f"{type(exc).__name__}"
    else:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                lines = {k: v for k, v in (l.split(":") for l in fh if ":" in l)}
            info["ram_total_gb"] = round(int(lines["MemTotal"].split()[0]) / 1e6, 1)
            info["ram_available_gb"] = round(int(lines["MemAvailable"].split()[0]) / 1e6, 1)
        except Exception as exc:  # noqa: BLE001
            info["ram_error"] = f"{type(exc).__name__}"
    return info


def probe_disk() -> dict:
    disks = {}
    if platform.system() == "Windows":
        for letter in "CDEFG":
            root = f"{letter}:\\"
            if os.path.exists(root):
                try:
                    usage = shutil.disk_usage(root)
                    disks[letter] = round(usage.free / 1e9, 1)
                except Exception:
                    pass
    else:
        for root in ("/", "/home", "/mnt", "/data"):
            if os.path.exists(root):
                try:
                    disks[root] = round(shutil.disk_usage(root).free / 1e9, 1)
                except Exception:
                    pass
    return {"free_gb": disks}


def probe_network() -> dict:
    hostname = socket.gethostname()
    ips: list[str] = []
    try:
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addr = item[4][0]
            if addr not in ips and not addr.startswith("127."):
                ips.append(addr)
    except Exception:
        pass
    # 兜底：连一个外部地址看默认出口 IP（不实际发包也能拿到本地地址）
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            addr = sock.getsockname()[0]
            if addr not in ips:
                ips.append(addr)
    except Exception:
        pass
    return {"hostname": hostname, "ipv4": ips}


# ---------------------------------------------------------------- GPU / 软件

def probe_gpu() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"present": False, "reason": "找不到 nvidia-smi（无 NVIDIA 驱动或非 NVIDIA 机器）"}
    raw = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        vram_mb = float(parts[2]) if parts[2].replace(".", "").isdigit() else 0.0
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "vram_total_gb": round(vram_mb / 1024, 1),
            "vram_used_gb": round(float(parts[3]) / 1024, 1) if parts[3].isdigit() else None,
            "driver": parts[4],
            "compute_cap": parts[5],
        })
    return {"present": bool(gpus), "gpus": gpus, "raw": raw.splitlines()}


def probe_torch() -> dict:
    info: dict = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_build"] = getattr(torch.version, "cuda", None)
        info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "total_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    info["nvcc"] = run(["nvcc", "--version"]).splitlines()[-1] if shutil.which("nvcc") else None
    info["cuda_toolkit_dirs"] = [
        d for d in (
            os.environ.get("CUDA_HOME"),
            "/usr/local/cuda",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        )
        if d and os.path.exists(d)
    ]
    info["vllm_installed"] = _module_version("vllm")
    info["lmcache_installed"] = _module_version("lmcache")
    return info


def _module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "已安装(无版本号)")
    except Exception:
        return None


def arch_capabilities(capability: list[int] | None) -> dict:
    """按 vLLM 的架构门槛判定这台机器能用哪些路径（见另一份分析文档 §3.3）。"""
    if not capability:
        return {"note": "无 CUDA 设备，只能用 CPU 路径"}
    major, minor = capability[0], capability[1]
    cap = major * 10 + minor
    return {
        "compute_capability": f"{major}.{minor}",
        "flash_attention_2": cap >= 80,
        "flash_attention_3": major == 9,
        "flash_attention_4": major >= 10,
        "flashinfer": 80 <= cap <= 121,
        "fp8_cutlass": cap >= 89,
        "fp8_marlin": cap in (89, 120, 121),
        "marlin_int4_w4a16": cap >= 75,
        "machete": cap >= 90,
        "verdict_attention": (
            "FA3（Hopper 专属）" if major == 9
            else "FA4（Blackwell）" if major >= 10
            else "FA2 + Triton + FlashInfer" if cap >= 80
            else "仅 Triton（FA2/FlashInfer 需 ≥8.0）"
        ),
        "verdict_quant": (
            "可试 FP8(CUTLASS/Marlin) + int4(W4A16)" if cap >= 89
            else "int4(W4A16/W8A16) + int8；FP8 基本不可用" if cap >= 80
            else "int4 Marlin (≥75) / int8"
        ),
    }


def recommend_dtype(gpu: dict) -> str:
    cap = gpu.get("compute_cap")
    if not cap:
        return "float32（CPU）"
    major, minor = (int(x) for x in cap.split("."))
    return "bfloat16" if (major, minor) >= (8, 0) else "float16"


def main() -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", default=socket.gethostname())
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    gpu = probe_gpu()
    torch_info = probe_torch()
    first = (gpu.get("gpus") or [{}])[0]
    caps = arch_capabilities(
        (torch_info.get("devices") or [{}])[0].get("capability")
        if torch_info.get("devices")
        else ([int(x) for x in str(first.get("compute_cap", "")).split(".")]
              if first.get("compute_cap") else None)
    )

    report = {
        "node_id": args.node_id,
        "suggested_agent_port": args.port,
        "os": probe_os(),
        "cpu_mem": probe_cpu_mem(),
        "disk": probe_disk(),
        "network": probe_network(),
        "gpu": gpu,
        "torch": torch_info,
        "arch_capabilities": caps,
        "recommended_dtype": recommend_dtype(first),
    }

    print("=" * 68)
    print(f"节点 {report['node_id']}  建议 agent 端口 {args.port}")
    print("=" * 68)
    os_info = report["os"]
    wsl = "（在 WSL 里）" if os_info.get("in_wsl") else ""
    print(f"[系统] {os_info['system']} {os_info['release']} {wsl}  build={os_info.get('windows_build','')}")
    if os_info.get("wsl"):
        print(f"        WSL: {os_info['wsl'].splitlines()[0] if os_info['wsl'] else '未安装'}")
    cm = report["cpu_mem"]
    print(f"[CPU ] {cm.get('cpu_cores')} 逻辑核   RAM 总 {cm.get('ram_total_gb','?')}GB / 可用 {cm.get('ram_available_gb','?')}GB")
    print(f"[磁盘] 可用 " + "  ".join(f"{k}:{v}GB" for k, v in report["disk"]["free_gb"].items()))
    print(f"[网络] {report['network']['hostname']}  IP={report['network']['ipv4']}")

    if not gpu.get("present"):
        print(f"[GPU ] 无 —— {gpu.get('reason')}")
    else:
        for g in gpu["gpus"]:
            print(f"[GPU ] #{g['index']} {g['name']}  {g['vram_total_gb']}GB  驱动 {g['driver']}  SM {g['compute_cap']}")
        print(f"[能力] 计算能力 {caps.get('compute_capability')} → attention: {caps.get('verdict_attention')}")
        print(f"       量化: {caps.get('verdict_quant')}")
        print(f"       建议 dtype: {report['recommended_dtype']}")

    t = report["torch"]
    if t.get("torch"):
        print(f"[框架] torch {t['torch']} (cuda_build={t.get('cuda_build')})  "
              f"cuda_available={t.get('cuda_available')}  devices={t.get('device_count')}")
    else:
        print(f"[框架] torch 不可用：{t.get('torch_error')}")
    print(f"       transformers={t.get('transformers')}  vllm={t.get('vllm_installed')}  "
          f"lmcache={t.get('lmcache_installed')}")
    print(f"       nvcc={t.get('nvcc') or '未安装（跑 PyTorch 不需要；编译 vLLM/LMCache 扩展才需要）'}")

    out = args.json_out or f"env_{args.node_id}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}（把它和另三台的一起交给控制端，用来定切层与角色）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
