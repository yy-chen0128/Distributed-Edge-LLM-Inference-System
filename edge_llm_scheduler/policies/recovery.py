"""中断恢复策略：任务中断后如何重新执行。

已实现：
- TokenRecovery：token 级恢复（参考 SpotServe stateful recovery）——已提交的 token
  KV 保留，只重算未提交的部分。

原理：自回归生成中，前 progress_tokens 个 token 的 KV 已稳定写入缓存（因果性保证不再
重算）。中断后恢复时，从 progress_tokens 继续，而不是从头重跑——减少重算量 = 提升
"最终系统性能"（节点离开/故障时恢复更快）。

TODO（后续补充）：
- 跨节点重调度（原节点可能已离开，恢复到剩余节点）
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
    """默认：新建一个相同请求的新任务（重跑，从头开始）。"""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    async def recover(self, task: Task) -> Optional[Task]:
        retries = task._retries
        if retries >= self.max_retries:
            logger.warning(f"task {task.task_id} gave up after {retries} retries")
            return None
        retry = Task(
            task_id=str(uuid.uuid4()),
            request_id=task.request_id,
            node_id=task.node_id,
            layer_range=task.layer_range,
            kv_block_ids=task.kv_block_ids,
            status="pending",
        )
        retry._retries = retries + 1
        logger.info(f"recovering task {task.task_id} as {retry.task_id} (from scratch)")
        return retry


class TokenRecovery(RecoveryPolicy):
    """token 级恢复：从已提交的 progress_tokens 继续，只重算未提交部分。

    已提交的 KV 块（在 task.kv_block_ids 里）保留复用；新任务携带 progress_tokens，
    让引擎从该位置继续而非从头。

    resume_node_id: 恢复时的目标节点（默认原节点；若已离开由调度器重路由）。
    """

    def __init__(self, max_retries: int = 3, resume_node_id: Optional[str] = None) -> None:
        self.max_retries = max_retries
        self.resume_node_id = resume_node_id

    async def recover(self, task: Task) -> Optional[Task]:
        retries = task._retries
        if retries >= self.max_retries:
            logger.warning(f"task {task.task_id} gave up after {retries} retries")
            return None

        retry = Task(
            task_id=str(uuid.uuid4()),
            request_id=task.request_id,
            node_id=self.resume_node_id or task.node_id,
            layer_range=task.layer_range,
            kv_block_ids=task.kv_block_ids,   # 已提交的 KV 保留复用
            status="pending",
            progress_tokens=task.progress_tokens,  # 从这继续
        )
        retry._retries = retries + 1
        logger.info(
            f"token-recovering task {task.task_id} as {retry.task_id} "
            f"(resume from {task.progress_tokens} tokens)"
        )
        return retry
