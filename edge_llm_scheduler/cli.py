"""CLI 演示入口：起一个 mock 集群，跑通完整请求生命周期。

用法：
  python -m edge_llm_scheduler.cli [--nodes 3] [--requests 5]

验证框架端到端：节点注册 → 模型装载 → 请求 → 路由 → 执行 → 结果 + 节点事件。
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .config import SchedulerConfig
from .core import (
    Event, EventBus, EventType, InferenceRequest, ModelSpec, Node,
    NodeCapability, NodeRole, NodeState, RequestFlow, TaskScheduler,
)
from .core.node_manager import NodeManager
from .backends import MockEngine, MockKVStore, MockTransport

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("cli")


def make_node(node_id: str, mem_gb: int, flops: float, load: float) -> Node:
    return Node(
        node_id=node_id,
        capability=NodeCapability(
            compute_flops=flops,
            memory_total=mem_gb * 1024**3,
            bandwidth=100.0,
            supported_roles={NodeRole.GENERAL},
        ),
        state=NodeState(memory_free=int(mem_gb * 0.8 * 1024**3), load=load),
    )


async def demo(args) -> None:
    config = SchedulerConfig()
    bus = EventBus()
    bus.start()

    # 事件日志订阅
    async def on_event(ev: Event):
        logger.info(f"  [event] {ev.type.name} node={ev.node_id} req={ev.request_id}")

    for et in EventType:
        bus.subscribe(et, on_event)

    # 节点 + 引擎
    nm = NodeManager(event_bus=bus)
    for i in range(args.nodes):
        nid = f"node_{i}"
        await nm.register(make_node(nid, mem_gb=8 + i, flops=100 + i * 20, load=0.1 + 0.1 * i))

    storage = MockKVStore()
    transport = MockTransport(bandwidth_mbps=100.0, latency_ms=5.0)
    scheduler = TaskScheduler(
        node_manager=nm, storage=storage, transport=transport, event_bus=bus
    )
    for nid in [n.node_id for n in nm.all_nodes()]:
        scheduler.attach_engine(nid, MockEngine(nid, tokens_per_sec=10 + i))

    flow = RequestFlow(scheduler)

    # 跑几个请求
    for r in range(args.requests):
        req = InferenceRequest(
            request_id=f"req-{r}",
            prompt=f"hello world request {r}",
            max_tokens=32,
        )
        result = await flow.process(req)
        logger.info(f"  => {req.request_id}: {result.num_tokens} tokens, "
                    f"lat={result.latency_ms:.1f}ms, err={result.error}")

    # 演示节点离开 → 迁移
    logger.info("-- simulating node_0 leaving --")
    await nm.unregister("node_0", reason="demo")
    await asyncio.sleep(0.2)

    # 再来一个请求（验证离开后系统仍工作）
    req = InferenceRequest(request_id="req-after-leave", prompt="still works", max_tokens=16)
    result = await flow.process(req)
    logger.info(f"  => {req.request_id}: {result.num_tokens} tokens, err={result.error}")

    await bus.stop()
    logger.info("demo done")


def main() -> None:
    parser = argparse.ArgumentParser(description="edge LLM scheduler demo")
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--requests", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(demo(args))


if __name__ == "__main__":
    main()
