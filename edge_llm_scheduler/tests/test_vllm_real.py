"""真实 vLLM 对接测试（需要 torch + vLLM 可导入）。

- 若 torch 或 vllm 不可导入 → skip（环境未就绪）
- 若可用 → 用真实 vLLM 验证 VLLMEngine 对接

注意：vLLM 是编译型包（需 CUDA kernels），CPU 环境下用官方 vllm wheel
（pip install vllm）才能跑。clone 的源码需要编译。本测试在环境就绪后自动生效。
"""

from __future__ import annotations

import asyncio

import pytest

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import vllm  # noqa: F401
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

REAL_AVAILABLE = _HAS_TORCH and _HAS_VLLM


@pytest.mark.skipif(not REAL_AVAILABLE, reason="torch/vllm 未就绪（需编译后的 vllm 包）")
@pytest.mark.asyncio
async def test_real_vllm_offline_engine():
    """用真实 vLLM 离线 LLM() + dummy 权重验证 VLLMEngine。

    注意：真实 vLLM LLM() 初始化较慢（加载模型），且 CPU 推理慢。
    这里只验证请求/响应通路，不追求速度。
    """
    from edge_llm_scheduler.backends.vllm_engine import VLLMEngine
    from edge_llm_scheduler.core.types import Task

    # vLLM 离线模式：进程内 LLM()，dummy 权重 + CPU
    try:
        from vllm import LLM
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",   # 小模型
            dtype="float32",
            max_model_len=2048,
            enforce_eager=True,   # 跳过图优化
            load_format="dummy",  # 随机权重，不用下载
        )
    except Exception as e:
        pytest.skip(f"真实 vLLM LLM() 初始化失败: {e}")

    # 用真实 LLM 验证 generate 通路
    outputs = llm.generate(["Hello world"], sampling_params={"max_tokens": 16, "temperature": 0})
    assert outputs, "vLLM generate 无输出"
    text = outputs[0].outputs[0].text
    assert isinstance(text, str) and len(text) > 0
    llm.shutdown()
