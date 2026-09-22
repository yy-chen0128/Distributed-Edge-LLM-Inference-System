"""无 GPU 的分层推理模拟器。

这个后端不执行矩阵乘法，但严格模拟控制面真正关心的语义：

* 每个节点只执行自己的连续 layer_range；
* stage 按顺序消费上一个 stage 的 activation；
* 每个 stage 生成属于自己层区间的 KVBlock；
* KV block 带有稳定 token key、层区间和大小，可被查询、迁移和恢复。

它不是 vLLM 的替代品，而是用来在没有 GPU 时验证分层调度和状态生命周期。
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from ..core.stage_runtime import StagePrepareResult, StageRuntime
from ..core.types import ActivationEnvelope, GenerationResult, KVBlock, StageAssignment, Task


class LayeredMockEngine(StageRuntime):
    """按层区间模拟一个 pipeline stage。"""

    def __init__(
        self,
        node_id: str,
        total_layers: int,
        hidden_size: int = 1024,
        prefill_ms_per_token: float = 0.02,
        decode_ms_per_token: float = 0.2,
        failure_rate: float = 0.0,
        time_scale: float = 1.0,
    ) -> None:
        self._node_id = node_id
        self.total_layers = max(1, total_layers)
        self.hidden_size = hidden_size
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token
        self.failure_rate = failure_rate
        self.time_scale = max(0.0, time_scale)
        self.stage_calls = 0
        self._assignments: dict[int, StageAssignment] = {}
        self._active_epoch: int | None = None
        self.received_activations: list[ActivationEnvelope] = []

    @property
    def node_id(self) -> str:
        return self._node_id

    async def prepare_epoch(
        self,
        assignment: StageAssignment,
        model_source: str = "model-store",
    ) -> StagePrepareResult:
        if assignment.node_id != self.node_id:
            raise ValueError(
                f"assignment for {assignment.node_id} cannot be prepared by {self.node_id}"
            )
        start, end = assignment.layer_range
        if start < 0 or end <= start or end > self.total_layers:
            raise ValueError(f"invalid layer range for {self.node_id}: {assignment.layer_range}")
        self._assignments[assignment.pipeline_epoch] = assignment
        return StagePrepareResult(
            node_id=self.node_id,
            pipeline_epoch=assignment.pipeline_epoch,
            layer_range=assignment.layer_range,
            model_source=model_source,
        )

    async def activate_epoch(self, pipeline_epoch: int) -> None:
        if pipeline_epoch not in self._assignments:
            raise ValueError(f"epoch {pipeline_epoch} was not prepared on {self.node_id}")
        self._active_epoch = pipeline_epoch

    async def retire_epoch(self, pipeline_epoch: int) -> None:
        # 调用方必须先确认该 epoch 没有在途请求。对于不再承担新计划的节点，
        # 旧 epoch 可能仍是它唯一的 active 配置，因此这里允许它退役。
        if pipeline_epoch == self._active_epoch:
            self._active_epoch = None
        self._assignments.pop(pipeline_epoch, None)

    async def generate(self, task: Task) -> GenerationResult:
        import random

        start, end = task.layer_range or (0, self.total_layers)
        assignment = self._assignments.get(task.pipeline_epoch)
        if assignment is not None and assignment.layer_range != (start, end):
            raise ValueError(
                f"task layer range {(start, end)} does not match epoch {task.pipeline_epoch} "
                f"assignment {assignment.layer_range} on {self.node_id}"
            )
        if self._active_epoch is not None and task.pipeline_epoch != self._active_epoch:
            raise ValueError(
                f"task epoch {task.pipeline_epoch} is not active on {self.node_id}; "
                f"active={self._active_epoch}"
            )
        if task.stage_index > 0:
            if not isinstance(task.activation, ActivationEnvelope):
                raise ValueError(f"stage {task.stage_index} on {self.node_id} requires an activation")
            if task.activation.request_id != task.request_id:
                raise ValueError("activation request id does not match task")
            if task.activation.pipeline_epoch != task.pipeline_epoch:
                raise ValueError("activation epoch does not match task")
            if task.activation.destination_node != self.node_id:
                raise ValueError("activation was delivered to the wrong stage")
            self.received_activations.append(task.activation)
        layer_count = max(1, end - start)
        prompt_len = task.prompt_len or self._prompt_len(task.prompt)
        max_tokens = max(1, task.max_tokens)

        # 每个 stage 只承担其层比例的计算时间；后续 stage 仍需处理同一
        # prefill/decode 流程，只是不再重复“读取用户 prompt”。
        layer_ratio = layer_count / self.total_layers
        prefill_ms = prompt_len * self.prefill_ms_per_token * layer_ratio
        decode_ms = max_tokens * self.decode_ms_per_token * layer_ratio
        await asyncio.sleep((prefill_ms + decode_ms) * self.time_scale / 1000.0)

        if random.random() < self.failure_rate:
            raise RuntimeError(f"layered mock engine {self.node_id} simulated failure")

        self.stage_calls += 1
        block_hash = self._block_hash(task, start, end)
        kv_tokens = self._token_ids(task.prompt)
        kv_bytes = max(1, len(kv_tokens) * layer_count * 2 * 2)
        block = KVBlock(
            block_hash=block_hash,
            num_tokens=len(kv_tokens),
            byte_size=kv_bytes,
            layer_range=(start, end),
            data=f"kv:{task.request_id}:{start}:{end}".encode(),
            tokens=kv_tokens,
            prefill_time_ms=prefill_ms,
        )

        # Decode 的 activation 大小约为一个 token 的 hidden states；prefill 模拟则
        # 用完整 prompt 长度，使无线带宽成本会真实进入 CPU 实验的传输路径。
        activation_bytes = max(1, prompt_len * self.hidden_size * 2)
        activation_seed = hashlib.sha256(
            f"{task.request_id}|{task.pipeline_epoch}|{start}|{end}".encode()
        ).digest()
        activation_payload = (
            activation_seed * ((activation_bytes + len(activation_seed) - 1) // len(activation_seed))
        )[:activation_bytes]
        activation = ActivationEnvelope(
            request_id=task.request_id,
            pipeline_epoch=task.pipeline_epoch,
            stage_index=task.stage_index,
            layer_range=(start, end),
            source_node=self.node_id,
            destination_node=None,
            payload=activation_payload,
            metadata={"hidden_size": self.hidden_size, "logical_bytes": activation_bytes},
        )
        is_final = task.stage_index == task.stage_count - 1
        return GenerationResult(
            request_id=task.request_id,
            text=f"[layered mock output from {self.node_id}]" if is_final else "",
            num_tokens=max_tokens if is_final else 0,
            kv_block_ids=[block_hash],
            kv_blocks=[block],
            hit_tokens=task.hit_tokens if task.stage_index == 0 else 0,
            stage_index=task.stage_index,
            activation=activation,
            metadata={
                "node_id": self.node_id,
                "layer_range": (start, end),
                "is_final_stage": is_final,
            },
        )

    @staticmethod
    def _prompt_len(prompt: Any) -> int:
        if prompt is None:
            return 0
        return len(prompt)

    @staticmethod
    def _token_ids(prompt: Any) -> list[int]:
        if prompt is None:
            return []
        if isinstance(prompt, str):
            return [
                int(hashlib.sha256(ch.encode("utf-8")).hexdigest()[:8], 16)
                for ch in prompt
            ]
        return [int(token) for token in prompt]

    @staticmethod
    def _block_hash(task: Task, start: int, end: int) -> int:
        raw = f"{task.request_id}|{task.prompt!r}|{start}|{end}|{task.progress_tokens}"
        return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)
