"""从 HuggingFace 镜像直连下载模型文件（不依赖 huggingface_hub 的 etag 握手）。

背景：部分校园网/镜像站不返回 X-Repo-Commit 等元数据头，`huggingface_hub`
会报 FileMetadataError / LocalEntryNotFoundError。本脚本直接走 HTTP：
    1) GET {endpoint}/api/models/{repo}/tree/main  取文件清单
    2) GET {endpoint}/{repo}/resolve/main/{path}   流式下载

用法（四台笔记本上都可用）：
    python scripts/hf_mirror_download.py --repo Qwen/Qwen2.5-0.5B-Instruct \
        --dest project/.models/Qwen2.5-0.5B-Instruct
    python scripts/hf_mirror_download.py --repo Qwen/Qwen2.5-7B-Instruct --dest ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

# 默认跳过：重复权重格式、训练中间态、非必需的大文件
SKIP_SUFFIXES = (".bin", ".pth", ".pt", ".msgpack", ".h5", ".onnx", ".gguf")
SKIP_PREFIXES = ("optimizer", "scheduler", "rng_state", "training_args", "events.out")
KEEP_IF_PRESENT = ("config.json", "generation_config.json")

# 镜像站会拦默认的 "Python-urllib/3.x" UA（实测 403 Forbidden），必须伪装成浏览器。
# 也可以用 HF_TOKEN 环境变量带 Bearer（镜像对公开仓库通常不需要）。
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _request(url: str, timeout: float) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def list_repo(endpoint: str, repo: str, timeout: float) -> list[dict]:
    url = f"{endpoint}/api/models/{repo}/tree/main"
    with urllib.request.urlopen(_request(url, timeout), timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected tree payload for {repo}: {type(payload)}")
    return [entry for entry in payload if entry.get("type") == "file"]


def should_skip(path: str, has_safetensors: bool) -> bool:
    name = os.path.basename(path)
    if name.startswith(SKIP_PREFIXES):
        return True
    if name.endswith(SKIP_SUFFIXES):
        # 有 safetensors 时跳过 .bin（很多仓库两种都放，体积翻倍）
        return has_safetensors or not name.endswith(".bin")
    return False


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(endpoint: str, repo: str, remote_path: str, dest_file: str, timeout: float) -> float:
    url = f"{endpoint}/{repo}/resolve/main/{remote_path}"
    os.makedirs(os.path.dirname(dest_file) or ".", exist_ok=True)
    tmp = dest_file + ".part"
    started = time.perf_counter()
    with urllib.request.urlopen(_request(url, timeout), timeout=timeout) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100.0 / total
                print(f"\r    {remote_path}: {pct:5.1f}% ({human(done)}/{human(total)})", end="")
    print()
    os.replace(tmp, dest_file)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="例如 Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dest", required=True, help="本地目标目录")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--include", default="", help="逗号分隔的白名单子串（可选）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    files = list_repo(args.endpoint, args.repo, args.timeout)
    has_safetensors = any(f["path"].endswith(".safetensors") for f in files)
    include = [s.strip() for s in args.include.split(",") if s.strip()]

    wanted = []
    for entry in files:
        path = entry["path"]
        if include and not any(token in path for token in include):
            continue
        if should_skip(path, has_safetensors):
            continue
        wanted.append(entry)

    if not wanted:
        print("没有匹配到要下载的文件", file=sys.stderr)
        return 2

    total_bytes = sum(int(e.get("size") or 0) for e in wanted)
    print(f"仓库 {args.repo} → {args.dest}")
    print(f"待下载 {len(wanted)} 个文件，共 {human(total_bytes)}（镜像 {args.endpoint}）")

    grand = time.perf_counter()
    for entry in wanted:
        path = entry["path"]
        dest_file = os.path.join(args.dest, path)
        if os.path.exists(dest_file) and not args.force:
            print(f"  跳过已存在: {path} ({human(os.path.getsize(dest_file))})")
            continue
        took = download(args.endpoint, args.repo, path, dest_file, args.timeout)
        size = os.path.getsize(dest_file)
        speed = size / max(1e-6, took)
        print(f"  完成 {path}  {human(size)}  {took:.1f}s  {human(speed)}/s")

    print(f"全部完成，用时 {time.perf_counter() - grand:.1f}s → {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
