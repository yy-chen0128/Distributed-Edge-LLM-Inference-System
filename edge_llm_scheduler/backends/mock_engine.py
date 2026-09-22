"""Mock 推理引擎：无 GPU 也能模拟推理行为。

模拟：prefill（按时延）+ decode（每 token 时延）。可配置吞吐/时延。
用于验证框架逻辑（路由→下发→回收），不涉及真实矩阵运算。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid

from ..core.task_scheduler import Engine
from ..core.types import GenerationResult, KVBlock, Task

logger = logging.getLogger(__name__)


class MockEngine(Engine):
    def __init__(
        self,
        node_id: str,
        tokens_per_sec: float = 20.0,
        prefill_ms_per_token: float = 0.1,
        failure_rate: float = 0.0,
        lat_multiplier: float = 1.0,
    ) -> None:
        self.node_id = node_id
        self.tokens_per_sec = tokens_per_sec
        self.prefill_ms = prefill_ms_per_token
        self.failure_rate = failure_rate
        self.lat_multiplier = lat_multiplier
        self.total_generated = 0

    async def generate(self, task: Task) -> GenerationResult:
        # 模拟计算时延：prefill 阶段（一次性）+ decode 阶段（每 token）
        prompt_len = task.prompt_len or getattr(task, "_prompt_len", 32)
        prefill_ms = prompt_len * self.prefill_ms * self.lat_multiplier
        decode_ms = (1000.0 / self.tokens_per_sec) * self.lat_multiplier
        await asyncio.sleep((prefill_ms + decode_ms) / 1000.0)

        if random.random() < self.failure_rate:
            raise RuntimeError(f"mock engine {self.node_id} simulated failure")

        # 模拟输出 token 数
        out_tokens = max(1, min(task.max_tokens, prompt_len // 2 if prompt_len else 16))
        self.total_generated += out_tokens

        # 模拟产出 KV 块
        new_blocks = []
        for _ in range(max(1, out_tokens // 16)):
            new_blocks.append(int(uuid.uuid4().int & 0xFFFFFFFF))
        kv_blocks = [
            KVBlock(
                block_hash=bid,
                num_tokens=out_tokens,
                byte_size=max(1, out_tokens * 128),
                layer_range=task.layer_range,
                data=b"mock-kv",
            )
            for bid in new_blocks
        ]

        result = GenerationResult(
            request_id=task.request_id,
            text=f"[mock output from {self.node_id}]",
            num_tokens=out_tokens,
            kv_block_ids=new_blocks,
            hit_tokens=0,
            kv_blocks=kv_blocks,
        )
        logger.debug(f"mock engine {self.node_id} done: {out_tokens} tokens")
        return result
