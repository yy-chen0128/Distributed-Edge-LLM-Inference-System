"""请求生命周期编排。

把 TaskScheduler 的请求处理组织成完整生命周期：
到达 → 排队 → 路由 → 执行 → 结果回收 → 事件发布。
当前是薄封装，后续可在各阶段挂接钩子（计时、日志、指标统计）。
"""

from __future__ import annotations

import logging
import time

from .task_scheduler import TaskScheduler
from .types import GenerationResult, InferenceRequest

logger = logging.getLogger(__name__)


class RequestFlow:
    def __init__(self, scheduler: TaskScheduler) -> None:
        self.scheduler = scheduler

    async def process(self, req: InferenceRequest) -> GenerationResult:
        start = time.time()
        result = await self.scheduler.handle_request(req)
        result.latency_ms = (time.time() - start) * 1000
        logger.info(
            f"request {req.request_id} done: tokens={result.num_tokens} "
            f"hit={result.hit_tokens} latency={result.latency_ms:.1f}ms "
            f"err={result.error}"
        )
        return result
