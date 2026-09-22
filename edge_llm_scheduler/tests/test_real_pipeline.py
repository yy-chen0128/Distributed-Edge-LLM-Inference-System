"""真实分层流水线（agent / 控制器 / 链路工具）的快速测试。

这些测试不加载真实模型权重（除了路径存在时才做的冒烟检查），因此可以在无 GPU、
无模型的环境里跑，作为 CI 的一部分。
"""

from __future__ import annotations

import os
import socket
import threading
import time

import pytest

from edge_llm_scheduler.experiments import measure_link
from edge_llm_scheduler.experiments.run_real_pipeline import split_layers
from edge_llm_scheduler.experiments.verify_equivalence import split_layers as split_layers_verify

MODEL_DIR = os.path.join(".models", "Qwen2.5-0.5B-Instruct")


# ------------------------------------------------------------------ 切层

@pytest.mark.parametrize("total,weights", [
    (24, [1, 1, 1, 1]),
    (24, [3, 2, 1, 1]),
    (24, [12, 6, 3, 3]),
    (28, [1, 2, 3, 4]),
    (5, [1, 1, 1]),          # 层数少于节点数：每段至少 1 层，最后一段兜住剩余
    (24, [10.5, 0.5, 0.5, 0.5]),
])
def test_split_layers_covers_model_exactly(total, weights):
    ranges = split_layers(total, weights, [f"n{i}" for i in range(len(weights))])
    assert len(ranges) == len(weights)
    # 连续且完整覆盖
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total
    for (start, end), (next_start, _) in zip(ranges, ranges[1:]):
        assert end == next_start
        assert end > start, f"空区间 {start}..{end}"


def test_verify_equivalence_split_matches_controller_split():
    """两处切层实现必须给出一致的层区间（否则验证与运行会用不同计划）。"""
    for weights in ([1, 1, 1, 1], [3, 2, 1, 1], [12, 6, 3, 3]):
        assert split_layers(24, weights, ["a"] * len(weights)) == \
            split_layers_verify(24, len(weights), weights)


def test_split_layers_is_monotonic_in_weight():
    ranges = split_layers(24, [4, 1], ["big", "small"])
    big = ranges[0][1] - ranges[0][0]
    small = ranges[1][1] - ranges[1][0]
    assert big > small


# ------------------------------------------------------------------ 链路工具

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_measure_link_round_trip():
    port = _free_port()
    thread = threading.Thread(target=measure_link.serve, args=("127.0.0.1", port), daemon=True)
    thread.start()
    time.sleep(0.5)
    # 不用 pytest 的 tmp_path：受限环境里系统临时目录可能不可写
    out = os.path.join(".models", "_test_link_tmp.json")
    os.makedirs(".models", exist_ok=True)
    try:
        result = measure_link.client("127.0.0.1", port, payload_mb=4.0, rtt_count=5,
                                     json_out=out)
        assert result["rtt_ms"]["samples"] == 5
        assert result["rtt_ms"]["median"] >= 0
        assert result["throughput_MBps"] > 0
        assert os.path.exists(out)
    finally:
        if os.path.exists(out):
            os.remove(out)
    # 估算表必须覆盖常见模型规模
    keys = " ".join(result["activation_cost_estimates"])
    assert "7B decode" in keys and "fp16" in keys


# ------------------------------------------------------- 引擎（需要 torch）

torch = pytest.importorskip("torch", reason="真实 stage 运行时需要 PyTorch")


def test_hidden_serialization_round_trip():
    from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine

    engine = object.__new__(HFLayeredEngine)
    engine.wire_dtype = torch.float16
    engine.device = torch.device("cpu")
    engine.dtype = torch.float32
    engine.hidden_size = 8

    tensor = torch.randn(1, 5, 8)
    payload, meta = HFLayeredEngine.serialize_hidden(engine, tensor)
    assert meta["shape"] == [1, 5, 8]
    assert meta["dtype"] == "float16"
    assert len(payload) == 5 * 8 * 2
    restored, _ = HFLayeredEngine.deserialize_hidden(engine, payload, meta)
    assert restored.shape == tensor.shape
    assert restored.dtype == torch.float32
    # fp16 往返误差应在 1e-3 量级
    assert torch.allclose(restored, tensor, atol=2e-3)


def test_deserialize_rejects_wrong_size():
    from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine

    engine = object.__new__(HFLayeredEngine)
    engine.wire_dtype = torch.float16
    engine.device = torch.device("cpu")
    engine.dtype = torch.float32
    engine.hidden_size = 8
    with pytest.raises(ValueError):
        HFLayeredEngine.deserialize_hidden(engine, b"\x00" * 10,
                                           {"shape": [1, 5, 8], "dtype": "float16"})


@pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="本地没有下载小模型")
def test_engine_shard_footprint_is_partial():
    """每个 stage 只应驻留自己那段层：首段含 embedding，末段含 norm/输出头。"""
    import asyncio

    from edge_llm_scheduler.backends.hf_layered_engine import HFLayeredEngine
    from edge_llm_scheduler.core.types import StageAssignment

    engine = HFLayeredEngine("t0", MODEL_DIR, device="cpu", dtype="float32",
                             logger=lambda *_: None)
    assert engine.total_layers == 24
    asyncio.run(engine.prepare_epoch(StageAssignment(1, "t0", (6, 12))))
    asyncio.run(engine.activate_epoch(1))
    stage = engine._active_stage()
    assert stage.layer_range == (6, 12)
    assert stage.embed is None, "中间段不应持有 embedding"
    assert stage.norm is None, "中间段不应持有 norm"
    # 6 层参数 ≈ 6 × 14.9M × 4B ≈ 358MB，必须远小于整模型
    assert 300e6 < stage.param_bytes < 420e6
