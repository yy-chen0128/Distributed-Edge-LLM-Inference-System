"""中断恢复策略：任务中断后如何重新执行。

TODO（后续补充复杂逻辑）：
- 从已提交的 KV 继续（token 级恢复，参考 SpotServe stateful recovery）
- 重新调度到其他节点（原节点可能已离开）
- 部分重算（只重算未提交的 token）

默认实现 DefaultRecovery：重新调度到负载最轻的健康节点，重跑整个任务。
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from ..core.types import Task

logger = logging.getLogger(__name__)


class RecoveryPolicy(ABC):
    @abstractmethod
    async def recover(self, task: Task) -> Optional[Task]:
        """返回新的 Task（重新执行），或 None（放弃）。"""


class DefaultRecovery(RecoveryPolicy):
    """默认：新建一个相同请求的新任务（重跑）。

    TODO：替换为 token 级恢复 + 目标节点选择。
    """

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    async def recover(self, task: Task) -> Optional[Task]:
        # 简单计数重试：task 无重试计数时视为第 1 次
        retries = getattr(task, "_retries", 0)
        if retries >= self.max_retries:
            logger.warning(f"task {task.task_id} gave up after {retries} retries")
            return None
        retry = Task(
            task_id=str(uuid.uuid4()),
            request_id=task.request_id,
            node_id=task.node_id,   # 原节点（若还在）或由调度器重新路由
            layer_range=task.layer_range,
            kv_block_ids=task.kv_block_ids,
            status="pending",
        )
        retry._retries = retries + 1  # type: ignore[attr-defined]
        logger.info(f"recovering task {task.task_id} as {retry.task_id}")
        return retry
