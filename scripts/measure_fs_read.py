"""测文件的"冷读"吞吐：用 posix_fadvise(DONTNEED) 把该文件踢出 page cache 再读。

为什么需要它：判断"模型参数放 /mnt/d（Windows 盘）还是放 WSL 的 ext4"到底差多少。
普通 dd 会被 page cache 骗（我们刚跑完流水线，权重还在缓存里），所以必须真的丢缓存。
不需要 root：POSIX_FADV_DONTNEED 对自己有读权限的文件就有效。

用法：
    python scripts/measure_fs_read.py <path> [<path> ...] [--mb 512] [--rounds 3]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time


def cold_read(path: str, mb: int, rounds: int, drop: bool = True) -> dict:
    size = os.path.getsize(path)
    want = min(mb * (1 << 20), size)
    throughputs = []
    fadvise_ok = True
    for _ in range(rounds):
        fd = os.open(path, os.O_RDONLY)
        try:
            if drop:
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    fadvise_ok = False
            os.lseek(fd, 0, os.SEEK_SET)
            read = 0
            t0 = time.perf_counter()
            while read < want:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                read += len(chunk)
            dt = time.perf_counter() - t0
        finally:
            os.close(fd)
        throughputs.append((read / (1 << 20)) / dt)      # MiB/s
    return {
        "path": path,
        "size_mb": size / (1 << 20),
        "read_mb": want / (1 << 20),
        "median_mib_s": statistics.median(throughputs),
        "min_mib_s": min(throughputs),
        "max_mib_s": max(throughputs),
        "samples": [round(v, 1) for v in throughputs],
        "cache_dropped": fadvise_ok and drop,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--mb", type=int, default=512, help="每次读多少 MiB")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warm", action="store_true",
                        help="不丢 page cache（测热读；对比冷读看缓存能省多少）")
    args = parser.parse_args()

    print(f"{'path':<58} {'size':>7} {'read':>6} {'MiB/s (median)':>15} {'samples':>26}")
    for path in args.paths:
        if not os.path.isfile(path):
            print(f"{path:<58} MISSING")
            continue
        r = cold_read(path, args.mb, args.rounds, drop=not args.warm)
        note = "" if r["cache_dropped"] else "  [cache kept]"
        name = path if len(path) <= 56 else "..." + path[-53:]
        print(f"{name:<58} {r['size_mb']:>6.0f}M {r['read_mb']:>5.0f}M "
              f"{r['median_mib_s']:>15.1f} {str(r['samples']):>26}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
