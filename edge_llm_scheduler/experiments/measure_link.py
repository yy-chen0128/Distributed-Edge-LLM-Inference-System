"""节点间链路测量：真实 RTT 与吞吐（四台笔记本部署前必做的第一步）。

为什么必须实测：本项目选择 PP 而不是 TP 的前提就是"节点间链路带宽不足"，而这个
结论只能靠现场测量支撑——WiFi 5/6、有线千兆、跨房间穿墙的差距能达到一个数量级，
直接决定 layer 切分能切多细、activation 压缩到 fp16 是否值得。

用法（A 机做服务端）：
    python -m edge_llm_scheduler.experiments.measure_link --serve --port 9200

（B 机做客户端，测 A→B 与 B→A）：
    python -m edge_llm_scheduler.experiments.measure_link --host 192.168.1.23 --port 9200 \
        --payload-mb 64 --rtt-count 50 --json-out link_b_to_a.json

输出：RTT 中位数/最小值/抖动、单向吞吐（MB/s、Mbps），以及按模型 hidden size
估算的"每个 stage 边界传一次 activation 的耗时"，直接可用于切层决策。
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import struct
import threading
import time

_HEADER = struct.Struct(">I")
_MAGIC = b"PAIRLINK"


def _recv_exactly(sock: socket.socket, count: int) -> bytes:
    chunks = b""
    while len(chunks) < count:
        chunk = sock.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError("连接关闭")
        chunks += chunk
    return chunks


def serve(host: str, port: int) -> None:
    """服务端：回显数据（测 RTT），并接收大块数据（测下行吞吐）。"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(8)
    print(f"[serve] 监听 {host}:{port}，Ctrl+C 退出", flush=True)

    def handle(conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                tag = _recv_exactly(conn, len(_MAGIC))
                if tag != _MAGIC:
                    return
                kind = conn.recv(1) or b"?"   # 注意：不要解包成 int
                (size,) = _HEADER.unpack(_recv_exactly(conn, _HEADER.size))
                body = _recv_exactly(conn, size) if size else b""
                if kind == b"E":          # echo：把收到的原样发回
                    conn.sendall(_HEADER.pack(len(body)) + body)
                elif kind == b"S":        # sink：不回数据，只回执
                    conn.sendall(_HEADER.pack(0))
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def client(host: str, port: int, payload_mb: float, rtt_count: int,
           json_out: str = "") -> dict:
    # ---- RTT：小包往返
    rtts = []
    with socket.create_connection((host, port), timeout=30) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for _ in range(rtt_count):
            body = b"x" * 64
            started = time.perf_counter()
            sock.sendall(_MAGIC + b"E" + _HEADER.pack(len(body)) + body)
            (length,) = _HEADER.unpack(_recv_exactly(sock, _HEADER.size))
            _recv_exactly(sock, length)
            rtts.append((time.perf_counter() - started) * 1000.0)

        # ---- 吞吐：按 1MiB 分帧发送（帧大小不能超过 payload 缓冲，否则会少发）
        payload = b"y" * (1 << 20)
        total = int(payload_mb * (1 << 20))
        started = time.perf_counter()
        sent = 0
        per_chunk = 1 << 20
        while sent < total:
            chunk = min(per_chunk, total - sent)
            sock.sendall(_MAGIC + b"S" + _HEADER.pack(chunk) + payload[:chunk])
            sent += chunk
        (length,) = _HEADER.unpack(_recv_exactly(sock, _HEADER.size))
        _recv_exactly(sock, length)
        elapsed = time.perf_counter() - started

    mbps = total / elapsed / 1e6          # MB/s
    result = {
        "peer": f"{host}:{port}",
        "rtt_ms": {
            "min": round(min(rtts), 3),
            "median": round(statistics.median(rtts), 3),
            "max": round(max(rtts), 3),
            "stdev": round(statistics.pstdev(rtts), 3),
            "samples": len(rtts),
        },
        "throughput_MBps": round(mbps, 2),
        "throughput_Mbps": round(mbps * 8, 1),
        "payload_MB": round(total / 1e6, 1),
        "elapsed_s": round(elapsed, 3),
    }

    # ---- 按常见模型规模估算单次 activation 传输耗时
    estimates = {}
    for name, hidden, seq in (("0.5B prefill 64tok", 896, 64),
                              ("1.5B prefill 64tok", 1536, 64),
                              ("7B prefill 128tok", 3584, 128),
                              ("7B decode 1tok", 3584, 1),
                              ("32B prefill 128tok", 5120, 128)):
        for dtype, width in (("fp16", 2), ("fp32", 4)):
            nbytes = hidden * seq * width
            goodput = mbps * 1e6 * 0.75     # 实际可用约七成
            transfer_ms = nbytes / max(1.0, goodput) * 1000.0
            # PP 一次 stage 边界 = 传输 + 1 个 RTT 的协议开销
            estimates[f"{name} {dtype}"] = {
                "bytes": nbytes,
                "transfer_ms": round(transfer_ms, 3),
                "with_rtt_ms": round(transfer_ms + statistics.median(rtts), 3),
            }
    result["activation_cost_estimates"] = estimates
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if json_out:
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--payload-mb", type=float, default=64.0)
    parser.add_argument("--rtt-count", type=int, default=50)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    if args.serve:
        serve("0.0.0.0", args.port)
        return 0
    client(args.host, args.port, args.payload_mb, args.rtt_count, args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
