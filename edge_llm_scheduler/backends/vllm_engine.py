"""vLLM 引擎后端：通过 OpenAI API 驱动 vLLM。

模式参考 Preble 的 vllm_runtime.py：调度层不碰 vLLM 内部，纯 HTTP 驱动。
配合 LMCache 时用 --kv-transfer-config 注入 KV 连接器（见 lmcache_storage.py 文档）。

启动 vLLM 的命令（由上层 NodeManager 或外部编排执行）：
  vllm serve <model> --port <port> --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

from ..core.task_scheduler import Engine
from ..core.types import GenerationResult, Task

logger = logging.getLogger(__name__)


class VLLMEngine(Engine):
    """对接一个已启动的 vLLM OpenAI API server。"""

    def __init__(self, node_id: str, base_url: str, model: Optional[str] = None,
                 timeout_s: float = 120.0) -> None:
        if aiohttp is None:
            raise ImportError("aiohttp required for VLLMEngine: pip install aiohttp")
        self.node_id = node_id
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def generate(self, task: Task) -> GenerationResult:
        if task.stage_count > 1:
            raise RuntimeError(
                "VLLMEngine OpenAI adapter cannot execute a layer stage. "
                "The OpenAI endpoint represents a complete vLLM engine; use "
                "LayeredMockEngine for CPU simulation or a vLLM distributed "
                "worker/runtime for real pipeline parallelism."
            )
        prompt = task.prompt if task.prompt is not None else getattr(task, "_prompt_text", "")
        max_tokens = task.max_tokens if task.max_tokens != 64 else getattr(task, "_max_tokens", 64)
        payload = {
            "model": self.model or "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/v1/completions", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"vLLM error {resp.status}: {text[:200]}")
                data = await resp.json()

        text = data.get("choices", [{}])[0].get("text", "")
        num_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return GenerationResult(
            request_id=task.request_id,
            text=text,
            num_tokens=num_tokens,
            kv_block_ids=[],   # KV 块由 LMCache 管理，调度层不直接拿
        )

    async def get_status(self) -> dict:
        """查询引擎状态（vLLM /v1/models）。"""
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}/v1/models") as resp:
                return {"status": resp.status, "url": self.base_url}
