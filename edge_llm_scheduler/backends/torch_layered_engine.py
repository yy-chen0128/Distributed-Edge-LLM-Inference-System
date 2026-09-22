"""Torch-backed stage runtime for local pipeline validation.

This backend is intentionally small: it does not try to be a Transformer
implementation. Its job is to validate the real control/data contract with
numeric hidden-state tensors, explicit parameter residency, epoch activation,
and layer-local KV records.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

from ..core.stage_runtime import StagePrepareResult, StageRuntime
from ..core.types import ActivationEnvelope, GenerationResult, KVBlock, StageAssignment, Task


@dataclass
class _LayerWeights:
    weight: Any
    bias: Any


@dataclass
class _PreparedStage:
    assignment: StageAssignment
    model_source: str
    layers: dict[int, _LayerWeights]


class TorchLayeredEngine(StageRuntime):
    """A deterministic tensor stage runtime with explicit layer loading."""

    def __init__(
        self,
        node_id: str,
        total_layers: int,
        hidden_size: int = 32,
        vocab_size: int = 256,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.torch = self._import_torch()
        self.device = self._resolve_device(device)
        self._node_id = node_id
        self.total_layers = max(1, total_layers)
        self.hidden_size = max(1, hidden_size)
        self.vocab_size = max(2, vocab_size)
        self.seed = seed
        self.stage_calls = 0
        self.received_activations: list[ActivationEnvelope] = []
        self._prepared: dict[int, _PreparedStage] = {}
        self._active_epoch: int | None = None
        self._embedding = self._make_embedding()
        self._lm_head = self._make_lm_head()

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def active_epoch(self) -> int | None:
        return self._active_epoch

    @property
    def loaded_layer_ranges(self) -> dict[int, tuple[int, int]]:
        return {
            epoch: prepared.assignment.layer_range
            for epoch, prepared in self._prepared.items()
        }

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
        layers = {
            layer_id: self._make_layer(layer_id)
            for layer_id in range(start, end)
        }
        self._prepared[assignment.pipeline_epoch] = _PreparedStage(
            assignment=assignment,
            model_source=model_source,
            layers=layers,
        )
        return StagePrepareResult(
            node_id=self.node_id,
            pipeline_epoch=assignment.pipeline_epoch,
            layer_range=assignment.layer_range,
            model_source=model_source,
        )

    async def activate_epoch(self, pipeline_epoch: int) -> None:
        if pipeline_epoch not in self._prepared:
            raise ValueError(f"epoch {pipeline_epoch} was not prepared on {self.node_id}")
        self._active_epoch = pipeline_epoch

    async def retire_epoch(self, pipeline_epoch: int) -> None:
        if pipeline_epoch == self._active_epoch:
            self._active_epoch = None
        self._prepared.pop(pipeline_epoch, None)

    async def generate(self, task: Task) -> GenerationResult:
        prepared = self._prepared.get(task.pipeline_epoch)
        if prepared is None:
            raise ValueError(f"epoch {task.pipeline_epoch} is not prepared on {self.node_id}")
        if self._active_epoch != task.pipeline_epoch:
            raise ValueError(
                f"task epoch {task.pipeline_epoch} is not active on {self.node_id}; "
                f"active={self._active_epoch}"
            )
        if prepared.assignment.layer_range != task.layer_range:
            raise ValueError(
                f"task layer range {task.layer_range} does not match "
                f"{prepared.assignment.layer_range} on {self.node_id}"
            )

        if task.stage_index == 0:
            hidden = self._embed_prompt(task.prompt)
        else:
            activation = self._checked_activation(task)
            self.received_activations.append(activation)
            hidden = self._deserialize_hidden(activation.payload)

        with self.torch.inference_mode():
            for layer_id in sorted(prepared.layers):
                layer = prepared.layers[layer_id]
                hidden = self.torch.tanh(hidden.matmul(layer.weight.t()) + layer.bias)

        self.stage_calls += 1
        payload = self._serialize_hidden(hidden)
        start, end = prepared.assignment.layer_range
        activation = ActivationEnvelope(
            request_id=task.request_id,
            pipeline_epoch=task.pipeline_epoch,
            stage_index=task.stage_index,
            layer_range=(start, end),
            source_node=self.node_id,
            destination_node=None,
            payload=payload,
            metadata={
                "tensor_kind": "torch.hidden_states",
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "logical_bytes": hidden.nelement() * hidden.element_size(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "device": str(self.device),
            },
        )

        block_hash = self._block_hash(task, start, end, payload)
        tokens = self._token_ids(task.prompt)
        block = KVBlock(
            block_hash=block_hash,
            num_tokens=len(tokens),
            byte_size=max(1, len(tokens) * (end - start) * self.hidden_size * 4),
            layer_range=(start, end),
            data=payload[:4096],
            tokens=tokens,
            prefill_time_ms=float(len(tokens) * (end - start)),
        )
        is_final = task.stage_index == task.stage_count - 1
        return GenerationResult(
            request_id=task.request_id,
            text=self._decode(hidden, task.max_tokens) if is_final else "",
            num_tokens=max(1, task.max_tokens) if is_final else 0,
            kv_block_ids=[block_hash],
            kv_blocks=[block],
            hit_tokens=task.hit_tokens if task.stage_index == 0 else 0,
            stage_index=task.stage_index,
            activation=activation,
            metadata={
                "node_id": self.node_id,
                "layer_range": (start, end),
                "is_final_stage": is_final,
                "model_source": prepared.model_source,
            },
        )

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("TorchLayeredEngine requires PyTorch") from exc
        return torch

    def _resolve_device(self, device: str) -> Any:
        normalized = device.lower()
        if normalized == "auto":
            normalized = "npu" if self._npu_is_available() else "cpu"
        if normalized.startswith("npu"):
            try:
                import torch_npu  # noqa: F401
            except ImportError as exc:  # pragma: no cover - requires Ascend env
                raise RuntimeError("device='npu' requires torch-npu") from exc
            if not self._npu_is_available():  # pragma: no cover - requires Ascend env
                raise RuntimeError("torch-npu is installed but no Ascend NPU is available")
        return self.torch.device(normalized)

    def _npu_is_available(self) -> bool:
        return bool(
            hasattr(self.torch, "npu")
            and hasattr(self.torch.npu, "is_available")
            and self.torch.npu.is_available()
        )

    def _make_generator(self, offset: int) -> Any:
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(self.seed + offset)
        return generator

    def _make_embedding(self) -> Any:
        generator = self._make_generator(10_000)
        return self.torch.randn(
            self.vocab_size,
            self.hidden_size,
            generator=generator,
            dtype=self.torch.float32,
        ).to(self.device)

    def _make_lm_head(self) -> Any:
        generator = self._make_generator(20_000)
        return self.torch.randn(
            self.vocab_size,
            self.hidden_size,
            generator=generator,
            dtype=self.torch.float32,
        ).to(self.device)

    def _make_layer(self, layer_id: int) -> _LayerWeights:
        generator = self._make_generator(layer_id + 1)
        weight = self.torch.randn(
            self.hidden_size,
            self.hidden_size,
            generator=generator,
            dtype=self.torch.float32,
        ) / (self.hidden_size ** 0.5)
        bias = self.torch.randn(
            self.hidden_size,
            generator=generator,
            dtype=self.torch.float32,
        ) / self.hidden_size
        return _LayerWeights(
            weight=weight.to(self.device).contiguous(),
            bias=bias.to(self.device).contiguous(),
        )

    def _embed_prompt(self, prompt: Any) -> Any:
        token_ids = self._token_ids(prompt) or [0]
        ids = self.torch.tensor(
            [token_id % self.vocab_size for token_id in token_ids],
            dtype=self.torch.long,
            device=self.device,
        )
        return self._embedding.index_select(0, ids).contiguous()

    def _checked_activation(self, task: Task) -> ActivationEnvelope:
        if not isinstance(task.activation, ActivationEnvelope):
            raise ValueError(f"stage {task.stage_index} on {self.node_id} requires an activation")
        activation = task.activation
        if activation.request_id != task.request_id:
            raise ValueError("activation request id does not match task")
        if activation.pipeline_epoch != task.pipeline_epoch:
            raise ValueError("activation epoch does not match task")
        if activation.destination_node != self.node_id:
            raise ValueError("activation was delivered to the wrong stage")
        if activation.metadata.get("tensor_kind") != "torch.hidden_states":
            raise ValueError("activation payload is not a torch hidden-state tensor")
        payload_sha = activation.metadata.get("payload_sha256")
        if payload_sha and payload_sha != hashlib.sha256(activation.payload).hexdigest():
            raise ValueError("activation payload checksum mismatch")
        return activation

    def _serialize_hidden(self, hidden: Any) -> bytes:
        buffer = io.BytesIO()
        self.torch.save({"hidden_states": hidden.detach().cpu().contiguous()}, buffer)
        return buffer.getvalue()

    def _deserialize_hidden(self, payload: bytes) -> Any:
        buffer = io.BytesIO(payload)
        try:
            body = self.torch.load(buffer, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - for older torch
            buffer.seek(0)
            body = self.torch.load(buffer, map_location="cpu")
        hidden = body.get("hidden_states") if isinstance(body, dict) else None
        if hidden is None:
            raise ValueError("activation tensor payload is missing hidden_states")
        if hidden.dim() != 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError(
                f"invalid hidden-state shape {tuple(hidden.shape)} for hidden_size={self.hidden_size}"
            )
        return hidden.to(device=self.device, dtype=self.torch.float32).contiguous()

    def _decode(self, hidden: Any, max_tokens: int) -> str:
        token_count = max(1, min(max_tokens, self.vocab_size))
        scores = hidden[-1].matmul(self._lm_head.t())
        _, token_ids = self.torch.topk(scores, k=token_count)
        return " ".join(f"tok{int(token)}" for token in token_ids.detach().cpu().tolist())

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
    def _block_hash(task: Task, start: int, end: int, payload: bytes) -> int:
        raw = (
            f"{task.request_id}|{task.pipeline_epoch}|{start}|{end}|"
            f"{hashlib.sha256(payload).hexdigest()}"
        )
        return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)
