"""回归测试：策略与调度器缺陷修复。

对应 `docs/research/policies-principles-and-testability.md` §7 的缺陷清单。
每个测试都对应一条曾经真实存在的行为，注释里写明"修之前是什么样"。
"""

from __future__ import annotations

import pytest

from edge_llm_scheduler.backends import MockKVStore
from edge_llm_scheduler.core import EventBus, TaskScheduler
from edge_llm_scheduler.core.node_manager import NodeManager
from edge_llm_scheduler.core.types import (
    Event,
    EventType,
    GenerationResult,
    InferenceRequest,
    KVBlock,
    ModelPlacement,
    Node,
    NodeCapability,
    NodeState,
    Task,
)
from edge_llm_scheduler.policies.migration import PriorityMigration
from edge_llm_scheduler.policies.placement import E2Placement, LayeredPipelinePlacement
from edge_llm_scheduler.policies.recovery import TokenRecovery


def make_node(node_id: str, flops: float = 100.0, load: float = 0.1,
              mem_free: int = 10 * 1024 ** 3) -> Node:
    return Node(
        node_id=node_id,
        capability=NodeCapability(compute_flops=flops, memory_total=16 * 1024 ** 3,
                                  bandwidth=100.0),
        state=NodeState(memory_free=mem_free, load=load),
    )


def make_block(h: int, reuse: int, prefill_ms: float, size: int = 1000) -> KVBlock:
    return KVBlock(block_hash=h, num_tokens=16, byte_size=size,
                   reuse_count=reuse, prefill_time_ms=prefill_ms)


def make_hashes() -> list:
    """与 TaskScheduler._hashes_from_prompt 一致的块哈希（用于制造命中）。"""
    tokens = list(range(32))
    return [hash(tuple(tokens[i:i + 16])) for i in range(0, len(tokens), 16)]


# ---------------------------------------------------------------- 缺陷 #1

@pytest.mark.asyncio
async def test_e2_without_prompt_does_not_crash_on_external_hit():
    """#1：prompt=None 且调用方传入 hit_tokens>0 时曾抛 UnboundLocalError。

    修前：`prefix_hashes` 只在 `prompt is not None` 分支里赋值，但 exploit 判据
    只要求 store 存在 → 命中数由外部传入时读未绑定变量，直接崩溃。
    """
    storage = MockKVStore()
    policy = E2Placement(storage=storage)
    nodes = [make_node("n1", load=0.1), make_node("n2", load=0.2)]
    req = InferenceRequest(request_id="r1", prompt=None, max_tokens=4)

    tasks = await policy.place(model=None, nodes=nodes, request=req,
                               storage=storage, hit_tokens=32)

    assert tasks and tasks[0].node_id in {"n1", "n2"}


# ---------------------------------------------------------------- 缺陷 #3

@pytest.mark.asyncio
async def test_priority_migration_spreads_blocks_over_targets():
    """#3：所有块曾落到同一个目标节点（静态 load 并列时 min 取列表第一个）。"""
    storage = MockKVStore()
    for h in range(4):
        await storage.save(make_block(h, reuse=5, prefill_ms=10.0), "leave:gpu")

    # 两个同等负载、显存有限的目标 → 记了账才会摊开
    nodes = [make_node("s1", mem_free=1 * 1024 ** 2),
             make_node("s2", mem_free=1 * 1024 ** 2)]
    plan = await PriorityMigration(deadline_ms=float("inf")).decide(
        "leave", [0, 1, 2, 3], storage, available_nodes=nodes)

    targets = [mv["dst_location"] for mv in plan.block_moves]
    assert set(targets) == {"s1:gpu", "s2:gpu"}, targets
    assert targets.count("s1:gpu") == 2 and targets.count("s2:gpu") == 2


# ---------------------------------------------------------------- 缺陷 #4

@pytest.mark.asyncio
async def test_move_cost_distinguishes_local_remote_and_slow_links():
    """#4：成本模型曾忽略 src/dst，本机与跨机同价。"""
    storage = MockKVStore()
    block = KVBlock(block_hash=1, num_tokens=16, byte_size=1_000_000)

    local = await storage.estimate_move_cost(block, "n1:gpu", "n1:cpu")
    remote = await storage.estimate_move_cost(block, "n1:gpu", "n2:gpu")
    storage.link_bandwidth_mbps["n3"] = 1.0        # 慢链路（1 MB/s）
    slow = await storage.estimate_move_cost(block, "n1:gpu", "n3:gpu")

    assert local < remote, (local, remote)
    assert remote < slow, (remote, slow)
    # 1MB / 100MB/s = 10ms 传输 + 跨机 RTT
    assert remote == pytest.approx(10.0 + storage.remote_rtt_ms)
    assert local == pytest.approx(10.0 + storage.local_rtt_ms)


# ---------------------------------------------------------------- 缺陷 #5

@pytest.mark.asyncio
async def test_expensive_but_never_reused_block_is_kept():
    """#5：reuse_count=0 的块优先级曾恒为 0，再贵也最先被丢。"""
    storage = MockKVStore()
    await storage.save(make_block(1, reuse=0, prefill_ms=99.0), "leave:gpu")  # 贵、没复用
    await storage.save(make_block(2, reuse=9, prefill_ms=10.0), "leave:gpu")  # 热、便宜
    nodes = [make_node("s1")]

    one_block_ms = await storage.estimate_move_cost(
        make_block(1, 0, 99.0), "leave:gpu", "s1:gpu")
    deadline = one_block_ms * 1.5      # 预算只够 1 块

    keep_expensive = await PriorityMigration(deadline_ms=deadline).decide(
        "leave", [1, 2], storage, available_nodes=nodes)
    assert [mv["block_hash"] for mv in keep_expensive.block_moves] == [1]

    # min_reuse_count=0 可退回原公式（此时保留的是"热但便宜"的那块）
    original = await PriorityMigration(deadline_ms=deadline,
                                       min_reuse_count=0).decide(
        "leave", [1, 2], storage, available_nodes=nodes)
    assert [mv["block_hash"] for mv in original.block_moves] == [2]


# ---------------------------------------------------------------- 缺陷 #8

@pytest.mark.asyncio
async def test_route_request_sets_hit_tokens_only_on_first_stage():
    """#8：调度器曾把 hit_tokens 覆盖到**所有**段，抹掉"只给首段"的规则。"""
    bus = EventBus()
    bus.start()
    nodes = NodeManager()
    for node_id in ("a", "b"):
        await nodes.register(make_node(node_id))

    storage = MockKVStore()
    policy = LayeredPipelinePlacement(placements=[
        ModelPlacement(node_id="a", layer_range=(0, 4)),
        ModelPlacement(node_id="b", layer_range=(4, 8)),
    ])
    scheduler = TaskScheduler(node_manager=nodes, storage=storage, event_bus=bus,
                              placement_policy=policy)

    prompt = list(range(32))
    for h in make_hashes():
        await storage.save(KVBlock(block_hash=h, num_tokens=16, byte_size=1000), "a:gpu")

    tasks = await scheduler.route_request(
        InferenceRequest(request_id="r1", prompt=prompt, max_tokens=4))

    assert len(tasks) == 2
    assert tasks[0].hit_tokens > 0
    assert tasks[1].hit_tokens == 0
    await bus.stop()


# ---------------------------------------------------------------- 缺陷 #7

class FlakyEngine:
    """第一次调用失败，之后成功：模拟"中断 + 恢复"。"""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.calls = 0

    async def generate(self, task: Task) -> GenerationResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated interruption")
        return GenerationResult(request_id=task.request_id, text="recovered",
                                num_tokens=1, stage_index=task.stage_index)


@pytest.mark.asyncio
async def test_node_leave_keeps_the_recovered_result():
    """#7：恢复出来的结果曾被丢弃，调用方永远拿不到。"""
    bus = EventBus()
    bus.start()
    nodes = NodeManager()
    node = make_node("a")
    await nodes.register(node)

    scheduler = TaskScheduler(node_manager=nodes, storage=MockKVStore(),
                              event_bus=bus,
                              recovery_policy=TokenRecovery(max_retries=2))
    engine = FlakyEngine("a")
    scheduler.attach_engine("a", engine)

    task = Task(task_id="t1", request_id="r1", node_id="a", layer_range=(0, 4))
    scheduler._tasks[task.task_id] = task
    node.state.pending_tasks.add(task.task_id)

    await scheduler._on_node_left(
        Event(type=EventType.NODE_LEFT, node_id="a", timestamp=0.0))

    assert engine.calls == 2                       # 首次失败 + 恢复一次
    assert "r1" in scheduler._results, "recovered result was dropped"
    assert scheduler._results["r1"].text == "recovered"
    await bus.stop()
