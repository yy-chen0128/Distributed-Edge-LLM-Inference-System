"""Probe the exact behaviour of the implemented policies (ASCII only).

Not a test suite: prints concrete evidence for the policy write-up.
"""
import asyncio
import sys

sys.path.insert(0, ".")

from edge_llm_scheduler.backends import MockKVStore
from edge_llm_scheduler.core.types import (
    InferenceRequest, KVBlock, ModelSpec, Node, NodeCapability, NodeRole,
    NodeState, Task,
)
from edge_llm_scheduler.policies.migration import DefaultMigration, PriorityMigration
from edge_llm_scheduler.policies.placement import E2Placement
from edge_llm_scheduler.policies.recovery import DefaultRecovery, TokenRecovery
from edge_llm_scheduler.policies.reparallelization import CapabilityReparallelization

MODEL = ModelSpec(model_name="demo", num_layers=12, hidden_size=1024,
                  total_params_bytes=384 * 1024 ** 2, kv_bytes_per_token=4096)


def node(nid, flops, load=0.0):
    return Node(
        node_id=nid,
        capability=NodeCapability(compute_flops=flops, memory_total=16 * 1024 ** 3,
                                  bandwidth=100.0, supported_roles={NodeRole.GENERAL}),
        state=NodeState(memory_free=8 * 1024 ** 3, load=load),
    )


async def main():
    print("=== [A] CapabilityReparallelization: layer split vs compute ===")
    pol = CapabilityReparallelization()
    for prof in ([100.0, 100.0, 100.0], [100.0, 150.0, 200.0], [20.0, 100.0, 400.0],
                 [50.0, 400.0], [100.0, 100.0, 100.0, 100.0]):
        nodes = [node(f"n{i}", f) for i, f in enumerate(prof)]
        pls = await pol.reconfigure(MODEL, nodes, {})
        print(f"  flops={prof} -> " + ", ".join(f"{p.node_id}{p.layer_range}" for p in pls))

    print()
    print("=== [B] PriorityMigration: value order + budget cutoff ===")
    store = MockKVStore()
    # byte_size 8192 -> cost = 8192/100*0.001 = 0.08192 ms per block
    specs = [(11, 5, 40.0), (12, 1, 40.0), (13, 9, 10.0), (14, 0, 99.0)]
    for bh, n_reuse, pt in specs:
        await store.save(KVBlock(block_hash=bh, num_tokens=16, byte_size=8192,
                                 reuse_count=n_reuse, prefill_time_ms=pt), "n0:gpu")
        print(f"  block {bh}: reuse_count={n_reuse} prefill_ms={pt} -> priority={n_reuse * pt}")
    for deadline in (float("inf"), 0.25, 0.10):
        pm = PriorityMigration(deadline_ms=deadline, bandwidth_mbps=100.0)
        plan = await pm.decide("n0", [11, 12, 13, 14], store,
                              [node("n1", 100.0), node("n2", 100.0)])
        moves = [(m["block_hash"], m["priority"], m["dst_location"]) for m in plan.block_moves]
        print(f"  deadline={deadline}: moves={moves} drops={plan.block_drops}")

    print()
    print("=== [C] Migration cost model and target choice ===")
    big = KVBlock(block_hash=99, num_tokens=16, byte_size=1_000_000)
    c = await store.estimate_move_cost(big, "n0:gpu", "n1:gpu")
    print(f"  1MB block at MockKVStore bandwidth 100MB/s -> {c:.3f} ms (src/dst ignored)")
    print("  -> target spread: every move above went to n1 (min static load); "
          "n2 got nothing")

    print()
    print("=== [D] Recovery: Default vs Token ===")
    t = Task(task_id="t1", request_id="r1", node_id="n0", layer_range=(0, 4),
             kv_block_ids=[7, 8], progress_tokens=25)
    r1 = await DefaultRecovery().recover(t)
    r2 = await TokenRecovery(max_retries=3).recover(t)
    print(f"  DefaultRecovery -> node={r1.node_id} kv={r1.kv_block_ids} "
          f"progress={r1.progress_tokens} retries={r1._retries}")
    print(f"  TokenRecovery   -> node={r2.node_id} kv={r2.kv_block_ids} "
          f"progress={r2.progress_tokens} retries={r2._retries}")
    t2 = Task(task_id="t2", request_id="r2", node_id="n0", _retries=3)
    print(f"  TokenRecovery(max_retries=3) on task with _retries=3 -> "
          f"{await TokenRecovery(max_retries=3).recover(t2)}")

    print()
    print("=== [E] E2Placement: prompt=None with external hit_tokens>0 ===")
    pol = E2Placement(storage=store)
    try:
        tasks = await pol.place(MODEL, [node("n1", 100.0), node("n2", 100.0)],
                                InferenceRequest(request_id="r9", prompt=None, max_tokens=4),
                                storage=store, hit_tokens=32)
        print(f"  no error; routed to {tasks[0].node_id}")
    except Exception as e:
        print(f"  RAISED {type(e).__name__}: {e}")


asyncio.run(main())
