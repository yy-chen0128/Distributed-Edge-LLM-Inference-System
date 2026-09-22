"""真实跨节点分层流水线推理（PP）控制器。

它做四件事，全部基于真实模型与真实网络：
1. **组集群**：把每台机器上的 `stage_agent` 当成一个节点，探查其能力（hello）。
2. **切层**：按各节点权重（可给不同算力/显存）把 24 层切成连续区间，作为一个
   pipeline epoch 下发（prepare → activate）。
3. **跑请求**：prefill 一次 + 自回归 decode N 步；每一步按 stage 顺序把 hidden
   states 通过 TCP 传给下一台机器，末段采样出 token 后再回到首段。采集每段
   计算耗时、排队耗时、activation 字节数、链路耗时。
4. **做实验**：
   - `--concurrency K`：K 个请求并发，观察流水线交叠带来的吞吐提升（PP bubble）。
   - `--leave-node N`：中途停掉一个节点，用剩余节点重新切层并切到新 epoch，
     验证"节点离开后系统继续服务"。

单机真实跑（本机起 4 个 agent 进程，走真实 TCP loopback）：
    python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 \
        --model .models/Qwen2.5-0.5B-Instruct --prompt "介绍一下你自己" --max-tokens 8

四台笔记本真实跑（每台先起 agent）：
    # 每台机器上：
    python -m edge_llm_scheduler.agents.stage_agent --node-id n0 --port 9100 \
        --model .models/Qwen2.5-0.5B-Instruct --device cuda:0
    # 控制端：
    python -m edge_llm_scheduler.experiments.run_real_pipeline \
        --cluster edge_llm_scheduler/deploy/cluster.example.json --max-tokens 16
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

_HEADER = struct.Struct(">I")


# ============================================================ 客户端

class AgentClient:
    """到某个 stage agent 的客户端。

    每个线程持有自己的长连接（thread-local），因此控制面可以对同一个 agent 并发
    下发请求——这样 agent 侧的排队（queue_ms）与"不同 stage 同时处理不同请求"的
    流水线交叠才是真实的。若共用一条连接，客户端锁会把并发抹平。
    """

    def __init__(self, node_id: str, host: str, port: int, timeout: float = 600.0) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.timeout = timeout
        self._local = threading.local()
        self._sockets: list[socket.socket] = []
        self._sockets_lock = threading.Lock()
        self._counter = itertools.count(1)

    # ---- 连接管理

    def _sock(self) -> socket.socket:
        sock = getattr(self._local, "sock", None)
        if sock is None:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._local.sock = sock
            with self._sockets_lock:
                self._sockets.append(sock)
        return sock

    def connect(self) -> None:
        self._sock()

    def close(self) -> None:
        current = getattr(self._local, "sock", None)
        if current is not None:
            try:
                current.close()
            finally:
                self._local.sock = None
        with self._sockets_lock:
            for sock in self._sockets:
                try:
                    sock.close()
                except OSError:
                    pass
            self._sockets.clear()

    # ---- 请求

    def call(self, op: str, payload: bytes = b"", **fields) -> tuple[dict, bytes]:
        request_id = next(self._counter)
        header = {"op": op, "id": request_id, **fields}
        body = json.dumps(header, ensure_ascii=False).encode("utf-8")
        frame = _HEADER.pack(len(body)) + body + _HEADER.pack(len(payload)) + payload
        try:
            sock = self._sock()
            sock.sendall(frame)
            response = self._read_frame(sock)
        except (ConnectionError, OSError):
            # 连接坏了就重连一次（节点重启/半开连接）
            self._local.sock = None
            sock = self._sock()
            sock.sendall(frame)
            response = self._read_frame(sock)
        if not response[0].get("ok"):
            raise RuntimeError(f"{self.node_id} 返回错误: {response[0].get('error')}")
        return response

    def _recv_exactly(self, sock: socket.socket, count: int) -> bytes:
        chunks = b""
        while len(chunks) < count:
            chunk = sock.recv(count - len(chunks))
            if not chunk:
                raise ConnectionError(f"{self.node_id} 连接已关闭")
            chunks += chunk
        return chunks

    def _read_frame(self, sock: socket.socket) -> tuple[dict, bytes]:
        (header_len,) = _HEADER.unpack(self._recv_exactly(sock, _HEADER.size))
        header = json.loads(self._recv_exactly(sock, header_len).decode("utf-8"))
        (payload_len,) = _HEADER.unpack(self._recv_exactly(sock, _HEADER.size))
        payload = self._recv_exactly(sock, payload_len) if payload_len else b""
        return header, payload


def split_layers(total: int, weights: list[float], names: list[str]) -> list[tuple[int, int]]:
    """按权重切连续层区间，保证完整覆盖且每段至少 1 层。"""
    stages = len(weights)
    total_w = sum(weights) or float(stages)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for idx, weight in enumerate(weights):
        remaining_stages = stages - idx - 1
        remaining_layers = total - cursor
        if remaining_stages == 0:
            end = total
        else:
            share = max(1, int(round(total * weight / total_w)))
            share = min(share, remaining_layers - remaining_stages)
            end = cursor + share
        ranges.append((cursor, end))
        cursor = end
    if ranges[-1][1] != total:
        start, _ = ranges[-1]
        ranges[-1] = (start, total)
    return ranges


# ============================================================ 控制器

@dataclass
class StageNode:
    node_id: str
    host: str
    port: int
    client: AgentClient
    device: str = "?"
    inventory: dict = field(default_factory=dict)
    weight: float = 1.0
    alive: bool = True
    layer_range: Optional[tuple] = None
    proc: Any = None                    # --local 模式下本机子进程


class RealPipeline:
    def __init__(self, stages: list[StageNode], model_path: str, dtype: str,
                 verbose: bool = True) -> None:
        self.stages = stages
        self.model_path = model_path
        self.dtype = dtype
        self.verbose = verbose
        self.epoch = 0
        self.records: list[dict] = []
        self.plan_history: list[dict] = []
        self._tokenizer = None

    # ------------------------------------------------------------ 分词与请求

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def encode(self, prompt: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return list(self.tokenizer(text)["input_ids"])

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # ---------------------------------------------------------------- 能力探查

    def probe(self) -> None:
        for stage in self.stages:
            header, _ = stage.client.call("hello")
            stage.device = f"{header.get('device_kind')}:{header.get('device')}"
            stage.inventory = header
            header.pop("ok", None)
            header.pop("id", None)
        if self.verbose:
            print("\n=== 集群盘点（各节点真实上报） ===")
            for stage in self.stages:
                inv = stage.inventory
                print(
                    f"  {stage.node_id:<6} {stage.host}:{stage.port}  device={inv.get('device')} "
                    f"dtype={inv.get('dtype')} layers={inv.get('total_layers')} "
                    f"hidden={inv.get('hidden_size')} vram={inv.get('vram_total_bytes', 0)/1e9:.1f}GB"
                )

    # ------------------------------------------------------------ 计划与 epoch

    def apply_plan(self, weights: Optional[list[float]] = None,
                   model_source: str = "model-store", reason: str = "initial") -> dict:
        alive = [s for s in self.stages if s.alive]
        if not alive:
            raise RuntimeError("没有存活节点")
        weights = weights or [s.weight for s in alive]
        total_layers = alive[0].inventory.get("total_layers") or 24
        ranges = split_layers(total_layers, weights, [s.node_id for s in alive])
        self.epoch += 1
        epoch = self.epoch

        started = time.perf_counter()
        prepare_ms: dict[str, float] = {}
        for stage, layer_range in zip(alive, ranges):
            t0 = time.perf_counter()
            stage.client.call("prepare_epoch", epoch=epoch, layer_range=list(layer_range),
                              model_source=model_source)
            prepare_ms[stage.node_id] = (time.perf_counter() - t0) * 1000.0
            stage.layer_range = layer_range
        for stage in alive:
            stage.client.call("activate_epoch", epoch=epoch)
        total_ms = (time.perf_counter() - started) * 1000.0

        plan = {
            "epoch": epoch,
            "reason": reason,
            "layers": {s.node_id: list(s.layer_range) for s in alive},
            "prepare_ms": prepare_ms,
            "switch_total_ms": total_ms,
            "nodes": [s.node_id for s in alive],
        }
        self.plan_history.append(plan)
        if self.verbose:
            print(f"\n=== pipeline epoch {epoch}（{reason}） ===")
            for stage in alive:
                rng = stage.layer_range
                print(f"  {stage.node_id:<6} layers {rng[0]:>2}..{rng[1]:<2} "
                      f"({rng[1]-rng[0]} 层)  prepare={prepare_ms[stage.node_id]:.0f}ms "
                      f"device={stage.host}")
            print(f"  切换总耗时 {total_ms:.0f}ms（先全量 prepare，再统一 activate）")
        return plan

    def retire_epoch(self, epoch: int) -> float:
        started = time.perf_counter()
        for stage in self.stages:
            if not stage.alive:
                continue
            try:
                stage.client.call("retire_epoch", epoch=epoch)
            except Exception:
                pass
        return (time.perf_counter() - started) * 1000.0

    # ------------------------------------------------------------------ 单请求

    def run_request(self, prompt: str, max_tokens: int, request_id: str) -> dict:
        alive = [s for s in self.stages if s.alive]
        if not alive:
            return {"request_id": request_id, "error": "no alive node"}
        tokens = self.encode(prompt)
        timeline: list[dict] = []
        started = time.perf_counter()

        # ---- prefill：首段吃 token，其余段吃上游 hidden
        hidden_payload = b""
        hidden_meta = None
        first_token = None
        for idx, stage in enumerate(alive):
            fields = {"request_id": request_id}
            payload = b""
            if idx == 0:
                fields["tokens"] = tokens
            else:
                payload = hidden_payload
                fields["hidden_meta"] = hidden_meta
            t0 = time.perf_counter()
            header, resp_payload = stage.client.call("prefill", payload=payload, **fields)
            hop_ms = (time.perf_counter() - t0) * 1000.0
            timeline.append({
                "phase": "prefill",
                "stage": stage.node_id,
                "layers": list(stage.layer_range),
                "compute_ms": header["compute_ms"],
                "queue_ms": header["queue_ms"],
                "hop_ms": round(hop_ms, 2),
                "activation_bytes": header.get("hidden_bytes", 0),
                "kv_bytes": header["metrics"].get("kv_bytes", 0),
                "token": header.get("token_id"),
            })
            hidden_payload = resp_payload
            hidden_meta = header.get("hidden_meta")
            if header.get("token_id") is not None:
                first_token = header["token_id"]
        generated = [first_token] if first_token is not None else []

        # ---- decode：每步再过一遍所有 stage
        prev_token = first_token
        for _ in range(max_tokens - 1):
            if prev_token is None:
                break
            hidden_payload = b""
            hidden_meta = None
            next_token = None
            for idx, stage in enumerate(alive):
                fields = {"request_id": request_id}
                payload = b""
                if idx == 0:
                    fields["token"] = int(prev_token)
                else:
                    payload = hidden_payload
                    fields["hidden_meta"] = hidden_meta
                t0 = time.perf_counter()
                header, resp_payload = stage.client.call("decode", payload=payload, **fields)
                hop_ms = (time.perf_counter() - t0) * 1000.0
                timeline.append({
                    "phase": "decode",
                    "stage": stage.node_id,
                    "layers": list(stage.layer_range),
                    "compute_ms": header["compute_ms"],
                    "queue_ms": header["queue_ms"],
                    "hop_ms": round(hop_ms, 2),
                    "activation_bytes": header.get("hidden_bytes", 0),
                    "kv_bytes": header["metrics"].get("kv_bytes", 0),
                    "token": header.get("token_id"),
                })
                hidden_payload = resp_payload
                hidden_meta = header.get("hidden_meta")
                if header.get("token_id") is not None:
                    next_token = header["token_id"]
            if next_token is None:
                break
            generated.append(next_token)
            prev_token = next_token

        total_ms = (time.perf_counter() - started) * 1000.0
        text = self.decode(generated) if generated else ""
        record = {
            "request_id": request_id,
            "epoch": self.epoch,
            "prompt_tokens": len(tokens),
            "prompt": prompt,
            "generated_tokens": generated,
            "generated_text": text,
            "total_ms": round(total_ms, 2),
            "ttft_ms": round(sum(s["hop_ms"] for s in timeline if s["phase"] == "prefill"), 2),
            "prefill_ms": round(sum(s["hop_ms"] for s in timeline if s["phase"] == "prefill"), 2),
            "decode_steps": max(0, len(generated) - 1),
            "per_stage": {},
            "timeline": timeline,
        }
        for phase in ("prefill", "decode"):
            for stage in alive:
                rows = [s for s in timeline if s["phase"] == phase and s["stage"] == stage.node_id]
                if not rows:
                    continue
                record["per_stage"].setdefault(stage.node_id, {})[phase] = {
                    "compute_ms_avg": round(sum(r["compute_ms"] for r in rows) / len(rows), 2),
                    "queue_ms_avg": round(sum(r["queue_ms"] for r in rows) / len(rows), 2),
                    "hop_ms_avg": round(sum(r["hop_ms"] for r in rows) / len(rows), 2),
                    "activation_bytes": rows[0]["activation_bytes"],
                    "kv_bytes_last": rows[-1]["kv_bytes"],
                }
        self.records.append(record)
        if self.verbose:
            self._print_request(record)
        return record

    def _print_request(self, record: dict) -> None:
        print(f"\n=== 请求 {record['request_id']}（epoch {record['epoch']}） ===")
        print(f"  prompt={record['prompt_tokens']} token → 生成 {len(record['generated_tokens'])} token")
        print(f"  总耗时 {record['total_ms']:.0f}ms  (prefill {record['prefill_ms']:.0f}ms)")
        print(f"  输出: {record['generated_text']}")
        print("  各段（prefill / decode 平均，单位 ms）:")
        for node_id, phases in record["per_stage"].items():
            parts = []
            for phase, stats in phases.items():
                parts.append(f"{phase}: compute={stats['compute_ms_avg']:.1f} "
                             f"queue={stats['queue_ms_avg']:.1f} hop={stats['hop_ms_avg']:.1f} "
                             f"act={stats['activation_bytes']/1024:.1f}KB")
            print(f"    {node_id:<6} " + " | ".join(parts))

    # ------------------------------------------------------------------ 并发实验

    def run_concurrent(self, prompt: str, max_tokens: int, concurrency: int) -> dict:
        """K 个请求并发：不同 stage 可以同时处理不同请求，流水线被填满。"""
        results: list[dict] = []
        lock = threading.Lock()

        def worker(idx: int) -> None:
            record = self.run_request(prompt, max_tokens, f"conc-{idx}")
            with lock:
                results.append(record)

        started = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall_ms = (time.perf_counter() - started) * 1000.0
        total_tokens = sum(len(r["generated_tokens"]) for r in results)
        summary = {
            "concurrency": concurrency,
            "wall_ms": round(wall_ms, 2),
            "requests": len(results),
            "generated_tokens": total_tokens,
            "throughput_tok_s": round(total_tokens / max(1e-6, wall_ms / 1000.0), 2),
            "avg_request_ms": round(sum(r["total_ms"] for r in results) / max(1, len(results)), 2),
        }
        if self.verbose:
            print(f"\n=== 并发实验 K={concurrency} ===")
            print(f"  墙钟 {summary['wall_ms']:.0f}ms  生成 {total_tokens} token  "
                  f"吞吐 {summary['throughput_tok_s']:.2f} tok/s  "
                  f"单请求平均 {summary['avg_request_ms']:.0f}ms")
        return summary


# ============================================================ 本地 agent 进程

def spawn_local_agents(count: int, model: str, dtype: str, ports: list[int],
                       device: str = "cpu", wire_dtype: str = "float16",
                       threads: int = 0, log_dir: str = ".models/logs") -> list[subprocess.Popen]:
    os.makedirs(log_dir, exist_ok=True)
    if threads <= 0:
        cores = os.cpu_count() or 4
        threads = max(1, cores // max(1, count))
    procs: list[subprocess.Popen] = []
    for idx in range(count):
        log_path = os.path.join(log_dir, f"agent_{idx}.log")
        handle = open(log_path, "w", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=".", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            [sys.executable, "-m", "edge_llm_scheduler.agents.stage_agent",
             "--node-id", f"n{idx}", "--port", str(ports[idx]),
             "--model", model, "--dtype", dtype, "--device", device,
             "--wire-dtype", wire_dtype, "--threads", str(threads)],
            stdout=handle, stderr=subprocess.STDOUT, env=env,
        )
        procs.append(proc)
    return procs


def wait_for_agents(clients: list[AgentClient], timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    for client in clients:
        while True:
            try:
                client.call("hello")
                break
            except (ConnectionError, OSError, RuntimeError):
                if time.time() > deadline:
                    raise TimeoutError(f"{client.node_id} 在 {timeout}s 内没有就绪")
                time.sleep(0.5)


# ============================================================ 入口

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=".models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local", type=int, default=0, help="本机起 N 个 agent 进程")
    parser.add_argument("--cluster", default="", help="四台笔记本的 cluster.json")
    parser.add_argument("--base-port", type=int, default=9100)
    parser.add_argument("--device", default="cpu", help="--local 模式下 agent 的设备")
    parser.add_argument("--wire-dtype", default="float16",
                        help="activation 传输 dtype（float16 省一半带宽；none 表示保持计算 dtype）")
    parser.add_argument("--threads", type=int, default=0,
                        help="每个 agent 进程的 torch 线程数（0=按核数/agent 数自动分摊）")
    parser.add_argument("--weights", default="", help="逗号分隔的切层权重，如 1,1,1,1")
    parser.add_argument("--prompt", default="用一句话解释什么是流水线并行。")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--concurrency", default="", help="逗号分隔的并发度，如 1,2,4")
    parser.add_argument("--leave-node", default="", help="中途停掉的节点 id（模拟节点离开）")
    parser.add_argument("--leave-after", type=int, default=1, help="第几个请求后触发离开")
    parser.add_argument("--metrics-out", default=".models/real_pipeline_metrics.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    procs: list[subprocess.Popen] = []
    stages: list[StageNode] = []

    if args.cluster:
        with open(args.cluster, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        model_path = config.get("model_path", args.model)
        entries = config["nodes"]
        for idx, entry in enumerate(entries):
            client = AgentClient(entry["node_id"], entry["host"], int(entry["port"]))
            stages.append(StageNode(entry["node_id"], entry["host"], int(entry["port"]),
                                    client, weight=float(entry.get("weight", 1.0))))
    else:
        count = args.local or 4
        ports = [args.base_port + i for i in range(count)]
        model_path = args.model
        procs = spawn_local_agents(count, model_path, args.dtype, ports,
                                   device=args.device, wire_dtype=args.wire_dtype,
                                   threads=args.threads)
        for idx, port in enumerate(ports):
            client = AgentClient(f"n{idx}", "127.0.0.1", port)
            stages.append(StageNode(f"n{idx}", "127.0.0.1", port, client))

    try:
        # 客户端按需建连，这里等 agent 的监听端口就绪即可
        wait_for_agents([s.client for s in stages])

        pipeline = RealPipeline(stages, model_path, args.dtype, verbose=verbose)
        pipeline.probe()

        weights = [float(w) for w in args.weights.split(",")] if args.weights else None
        if weights and len(weights) != len(stages):
            raise SystemExit(f"--weights 数量({len(weights)}) 与节点数({len(stages)}) 不一致")
        pipeline.apply_plan(weights, reason="initial")

        summary: dict = {"model": model_path, "dtype": args.dtype,
                         "wire_dtype": args.wire_dtype,
                         "nodes": [s.node_id for s in stages],
                         "inventory": [s.inventory for s in stages],
                         "plans": pipeline.plan_history}

        # 基础请求
        for idx in range(max(1, args.leave_after)):
            pipeline.run_request(args.prompt, args.max_tokens, f"req-{idx}")

        # 节点离开 → 重新切层 → 新 epoch
        if args.leave_node:
            leaving = next((s for s in stages if s.node_id == args.leave_node), None)
            if leaving is None:
                raise SystemExit(f"没有节点 {args.leave_node}")
            print(f"\n=== 模拟节点离开：{leaving.node_id} ===")
            try:
                leaving.client.call("shutdown")
            except Exception:
                pass
            leaving.client.close()
            leaving.alive = False
            if leaving.proc is not None:
                try:
                    leaving.proc.terminate()
                except Exception:
                    pass
            time.sleep(1.0)
            remaining = [s for s in stages if s.alive]
            print(f"  存活节点: {[s.node_id for s in remaining]}")
            reconfigure = pipeline.apply_plan(reason=f"{leaving.node_id} left")
            summary["node_left"] = {
                "left": leaving.node_id,
                "remaining": [s.node_id for s in remaining],
                "new_epoch": reconfigure["epoch"],
                "switch_total_ms": reconfigure["switch_total_ms"],
            }
            after = pipeline.run_request(args.prompt, args.max_tokens, "after-leave")
            summary["after_leave"] = {
                "ok": bool(after.get("generated_tokens")),
                "total_ms": after["total_ms"],
                "text": after["generated_text"],
            }
            # 排空旧 epoch（真实系统里由入口确认无在途请求后调用）
            summary["drain_old_epoch_ms"] = round(pipeline.retire_epoch(reconfigure["epoch"] - 1), 2)

        # 并发实验
        if args.concurrency:
            summary["concurrency"] = []
            for value in [int(v) for v in args.concurrency.split(",")]:
                summary["concurrency"].append(
                    pipeline.run_concurrent(args.prompt, args.max_tokens, value)
                )

        summary["requests"] = pipeline.records
        summary["finish"] = True
        if args.metrics_out:
            os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
            with open(args.metrics_out, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            print(f"\n指标已写入 {args.metrics_out}")
        return 0
    finally:
        for stage in stages:
            try:
                stage.client.close()
            except Exception:
                pass
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
