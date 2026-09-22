"""真实的 HuggingFace 分层 stage 运行时（替换 toy `TorchLayeredEngine`）。

与 `TorchLayeredEngine` 的区别：这里跑的是**真实预训练 Transformer**（真实
tokenizer、真实权重、真实 KV cache、真实 logits/采样），按层区间切分后每个
节点只驻留自己那几层的参数，stage 之间只传 hidden states（activation）。

实现要点（为什么这么写）：

1. **只加载本段权重**：先用 `torch.device("meta")` 建出模型骨架（不占内存），
   再从 safetensors 只取本 stage 需要的 key（`load_state_dict(assign=True)`）。
   这样一台笔记本不会先把整个模型读进内存——这正是"每节点只持有一部分层"
   的账本语义，也是 `ModelManager` 想管理的对象。
2. **逐层手写前向**：直接调用 `Qwen2DecoderLayer`，自己算 RoPE 的
   `position_embeddings` 并用 `transformers.masking_utils.create_causal_mask`
   构造因果掩码——与 HF 内部路径一致，所以分段执行的数值结果可与整体执行对齐。
3. **每 stage 自己的 KV**：`layer_idx` 在 stage 内重映射为 0..n-1，每个请求一份
   `DynamicCache`，只存本段的 K/V。因此 PD/PP 正常推进时不需跨节点搬 KV；
   只有节点离开、重分层时才需要搬（对应 `KVBlock.layer_range` 的部分迁移）。
4. **权重共享（tie_word_embeddings）**：末段需要 lm_head；若 checkpoint 里没有
   独立 lm_head（Qwen2.5 就是绑定的），末段必须额外持有一份 embed_tokens 并用它
   当输出头。这个额外开销会如实计入 `param_bytes`，不做隐藏。
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.stage_runtime import StagePrepareResult, StageRuntime
from ..core.types import ActivationEnvelope, GenerationResult, KVBlock, StageAssignment, Task

try:  # 可选依赖：只有真实运行才需要
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class StageForward:
    """一次 stage 前向的返回：回传给下一段的 hidden + 本段指标。"""

    hidden: Any = None                     # torch.Tensor，末段可为 None
    token_id: Optional[int] = None         # 只有末段给出下一 token
    logits_last: Any = None                # 末段最后一个位置的 logits（可选）
    metrics: dict = field(default_factory=dict)


@dataclass
class _PreparedStage:
    epoch: int
    layer_range: tuple
    layers: Any                            # nn.ModuleList（本地索引 0..n-1）
    embed: Any                             # 仅首段（或末段需要绑定输出头时）
    norm: Any                              # 仅末段
    lm_head: Any                           # 仅末段
    device: str
    param_bytes: int
    device_bytes: int


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


class HFLayeredEngine(StageRuntime):
    """一个节点上的真实分层 stage：只驻留 `layer_range` 内的解码层。"""

    def __init__(
        self,
        node_id: str,
        model_path: str,
        device: str = "cpu",
        dtype: str = "auto",
        max_seq_len: int = 4096,
        wire_dtype: str = "float16",
        logger=print,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("HFLayeredEngine 需要 PyTorch")
        self._node_id = node_id
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        # 传输用 dtype：链路是瓶颈时把 activation 压成 fp16（体积减半）。
        # None/"" 表示保持计算 dtype 不变（本机 loopback 时更省转换开销）。
        mapping = {"float16": torch.float16, "fp16": torch.float16,
                   "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                   "float32": torch.float32, "fp32": torch.float32}
        self.wire_dtype = mapping.get((wire_dtype or "").lower()) if wire_dtype else None
        self.max_seq_len = max_seq_len
        self.log = logger

        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(model_path)
        self.total_layers = int(self.config.num_hidden_layers)
        self.hidden_size = int(self.config.hidden_size)
        self.vocab_size = int(self.config.vocab_size)
        self.weight_files, self.has_lm_head_weight = self._weight_files(model_path)

        self._prepared: dict[int, _PreparedStage] = {}
        self._active_epoch: Optional[int] = None
        self._caches: dict[tuple[int, str], Any] = {}
        self.stage_calls = 0
        self.prepare_events: list[dict] = []

    # ------------------------------------------------------------------ 静态信息

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def active_epoch(self) -> Optional[int]:
        return self._active_epoch

    @property
    def decoded_layers(self) -> int:
        """本节点当前驻留的解码层数量。"""
        stage = self._active_stage()
        if stage is None:
            return 0
        start, end = stage.layer_range
        return end - start

    def inventory(self) -> dict:
        """给控制面看的节点能力清单（真实测量值，不是声明值）。"""
        return {
            "node_id": self.node_id,
            "model_path": self.model_path,
            "device": str(self.device),
            "device_kind": self.device.type,
            "dtype": str(self.dtype),
            "total_layers": self.total_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "has_lm_head_weight": self.has_lm_head_weight,
            "tie_word_embeddings": bool(getattr(self.config, "tie_word_embeddings", False)),
            "active_epoch": self._active_epoch,
            "prepared_epochs": sorted(self._prepared),
            "vram_total_bytes": self._device_total_bytes(),
            "param_bytes": (self._active_stage().param_bytes if self._active_stage() else 0),
            "device_bytes": (self._active_stage().device_bytes if self._active_stage() else 0),
        }

    # ------------------------------------------------------- StageRuntime 接口

    async def prepare_epoch(
        self,
        assignment: StageAssignment,
        model_source: str = "model-store",
    ) -> StagePrepareResult:
        start, end = assignment.layer_range
        if assignment.node_id != self.node_id:
            raise ValueError(f"{self.node_id} 不能准备给 {assignment.node_id} 的 stage")
        if start < 0 or end <= start or end > self.total_layers:
            raise ValueError(f"非法层区间 {assignment.layer_range}（模型共 {self.total_layers} 层）")

        t0 = time.perf_counter()
        model, owned = self._materialize_shard(start, end)
        load_ms = (time.perf_counter() - t0) * 1000.0

        # 按张量对象去重计入：绑定权重（lm_head.weight is embed_tokens.weight）
        # 只应算一份，否则会虚报驻留显存。
        seen: set[int] = set()
        param_bytes = 0
        for module in (owned["layers"], owned["embed"], owned["norm"], owned["lm_head"]):
            if module is None:
                continue
            for param in module.parameters():
                if id(param) in seen:
                    continue
                seen.add(id(param))
                param_bytes += param.numel() * param.element_size()
        device_bytes = self._device_allocated()

        self._prepared[assignment.pipeline_epoch] = _PreparedStage(
            epoch=assignment.pipeline_epoch,
            layer_range=(start, end),
            layers=owned["layers"],
            embed=owned["embed"],
            norm=owned["norm"],
            lm_head=owned["lm_head"],
            device=str(self.device),
            param_bytes=param_bytes,
            device_bytes=device_bytes,
        )
        event = {
            "node_id": self.node_id,
            "epoch": assignment.pipeline_epoch,
            "layer_range": [start, end],
            "layers": end - start,
            "param_bytes": param_bytes,
            "load_ms": load_ms,
            "device": str(self.device),
            "model_source": model_source,
        }
        self.prepare_events.append(event)
        self.log(
            f"[{self.node_id}] prepare epoch={assignment.pipeline_epoch} "
            f"layers={start}..{end} params={_human(param_bytes)} device={self.device} "
            f"load={load_ms:.0f}ms"
        )
        # 骨架模型对象本身不再需要（权重已在本段 ModuleList 里被引用）
        del model
        gc.collect()
        return StagePrepareResult(
            node_id=self.node_id,
            pipeline_epoch=assignment.pipeline_epoch,
            layer_range=(start, end),
            model_source=model_source,
        )

    async def activate_epoch(self, pipeline_epoch: int) -> None:
        if pipeline_epoch not in self._prepared:
            raise ValueError(f"epoch {pipeline_epoch} 未在本节点准备")
        self._active_epoch = pipeline_epoch

    async def retire_epoch(self, pipeline_epoch: int) -> None:
        stage = self._prepared.pop(pipeline_epoch, None)
        if pipeline_epoch == self._active_epoch:
            self._active_epoch = None
        for key in [k for k in self._caches if k[0] == pipeline_epoch]:
            self._caches.pop(key, None)
        if stage is not None:
            stage.layers = None
            stage.embed = None
            stage.norm = None
            stage.lm_head = None
        gc.collect()
        self._empty_device_cache()

    async def generate(self, task: Task) -> GenerationResult:
        """兼容框架 TaskScheduler 的单 stage 执行（真实 PP 由控制器驱动）。

        语义：stage_index==0 时用 task.prompt 的词表 id 做 embedding，否则要求
        task.activation 里带上一段的 hidden states。每调用一次推进一次前向，
        位置由 task.progress_tokens 决定（真实 decode 循环在控制器里）。
        """
        if task.pipeline_epoch != self._active_epoch:
            raise ValueError(
                f"任务 epoch {task.pipeline_epoch} 不是本节点活跃 epoch {self._active_epoch}"
            )
        stage = self._active_stage()
        assert stage is not None
        start, end = stage.layer_range

        payload_bytes = 0
        hidden = None
        if start == 0:
            tokens = self._token_ids(task.prompt)
            if task.progress_tokens > 0:
                if self._has_cache(task.request_id):
                    # 真续算：本进程还持有该请求的 past_key_values，
                    # 推进一个 token 即可（这才是 TokenRecovery 想省的算）。
                    tokens = tokens[-1:]
                else:
                    # 恢复目标换了节点、或引擎重启过 → 本地没有该请求的 KV。
                    # 此时只喂最后一个 token 会让模型看一个 1-token 序列，
                    # 输出相对原上下文是错的；退回整段重算：慢，但正确。
                    self.log(
                        f"[{self.node_id}] resume {task.request_id} has no local KV "
                        f"cache (progress_tokens={task.progress_tokens}); "
                        f"recomputing the full prompt"
                    )
            t0 = time.perf_counter()
            out = self._forward(tokens=None, token_ids=tokens, request_id=task.request_id)
        else:
            if not isinstance(task.activation, ActivationEnvelope):
                raise ValueError(f"stage {task.stage_index} 需要上一段的 activation")
            payload_bytes = len(task.activation.payload)
            hidden, meta = self.deserialize_hidden(task.activation.payload)
            t0 = time.perf_counter()
            out = self._forward(token_ids=None, hidden=hidden, request_id=task.request_id)
        compute_ms = (time.perf_counter() - t0) * 1000.0

        envelope = None
        if out.hidden is not None:
            payload, meta = self.serialize_hidden(out.hidden)
            envelope = ActivationEnvelope(
                request_id=task.request_id,
                pipeline_epoch=task.pipeline_epoch,
                stage_index=task.stage_index,
                layer_range=(start, end),
                source_node=self.node_id,
                destination_node=None,
                payload=payload,
                metadata={**meta, "compute_ms": compute_ms},
            )
        is_final = end == self.total_layers
        return GenerationResult(
            request_id=task.request_id,
            text=f"tok{out.token_id}" if (is_final and out.token_id is not None) else "",
            num_tokens=1 if is_final else 0,
            kv_block_ids=[],
            hit_tokens=task.hit_tokens if start == 0 else 0,
            stage_index=task.stage_index,
            activation=envelope,
            metadata={
                "node_id": self.node_id,
                "layer_range": (start, end),
                "compute_ms": compute_ms,
                "activation_bytes": len(envelope.payload) if envelope else 0,
                "received_bytes": payload_bytes,
                "token_id": out.token_id,
                "is_final_stage": is_final,
            },
        )

    async def shutdown(self) -> None:  # pragma: no cover - 由 agent 调用
        for epoch in list(self._prepared):
            await self.retire_epoch(epoch)

    # --------------------------------------------------------- 真实 PP 数据面

    def prefill(self, request_id: str, token_ids: Optional[list[int]] = None,
                hidden: Any = None) -> StageForward:
        """首段吃 prompt token，其余段吃上一段的 hidden states。"""
        stage = self._require_active()
        start, _ = stage.layer_range
        if start == 0:
            if token_ids is None:
                raise ValueError("首段 prefill 需要 token_ids")
            return self._forward(token_ids=list(token_ids), request_id=request_id)
        if hidden is None:
            raise ValueError(f"stage {start} 的 prefill 需要上游 hidden states")
        return self._forward(hidden=hidden, request_id=request_id)

    def decode(self, request_id: str, token_id: Optional[int] = None,
               hidden: Any = None) -> StageForward:
        """自回归一步：首段吃上一轮采样出的 token，其余段吃上游 hidden。"""
        stage = self._require_active()
        start, _ = stage.layer_range
        if start == 0:
            if token_id is None:
                raise ValueError("首段 decode 需要 token_id")
            return self._forward(token_ids=[int(token_id)], request_id=request_id)
        if hidden is None:
            raise ValueError(f"stage {start} 的 decode 需要上游 hidden states")
        return self._forward(hidden=hidden, request_id=request_id)

    def release_request(self, request_id: str) -> None:
        for key in [k for k in self._caches if k[1] == request_id]:
            self._caches.pop(key, None)

    def kv_bytes(self, request_id: str) -> int:
        """本请求在该 stage 的 KV 占用（字节）。

        两个坑（都实测过）：
        1. **不能写 `cache[i]`**：transformers 5.x 的 `DynamicCache` 不再可下标
           （抛 `TypeError: 'DynamicCache' object is not subscriptable`），旧写法被
           `except ... continue` 吞掉后**恒返回 0**——四机那次运行的
           `kv_bytes_last: 0` 就是这个原因。真实 KV 在 `cache.layers[i].keys/.values`。
        2. **不能用 `torch.cuda.memory_allocated` 的增量当 KV**：实测 6 层 38 token
           的真实 KV 只有 0.117 MB，而同一次调用的显存增量是 8.64 MB（74 倍），
           且释放请求后增量不降——那部分是前向中间张量与分配器池，不是 KV。

        逐张量求和即可：本 stage 只写自己那几个 slot，空 slot 贡献 0。
        """
        cache = self._caches.get((self._active_epoch or 0, request_id))
        if cache is None:
            return 0
        return sum(t.numel() * t.element_size()
                   for t in self._iter_cache_tensors(cache))

    @staticmethod
    def _iter_cache_tensors(cache):
        """跨 transformers 版本取出 KV 张量。

        transformers >= 5：``cache.layers[i].keys`` / ``.values``（`DynamicLayer`）
        transformers 4.4x：``cache.key_cache`` / ``cache.value_cache``（list of tensor）
        """
        layers = getattr(cache, "layers", None)
        if layers is not None:
            for layer in layers:
                for attr in ("keys", "values"):
                    tensor = getattr(layer, attr, None)
                    if tensor is not None and hasattr(tensor, "numel"):
                        yield tensor
            return
        for attr in ("key_cache", "value_cache"):
            for tensor in getattr(cache, attr, None) or []:
                if tensor is not None and hasattr(tensor, "numel"):
                    yield tensor

    def request_positions(self) -> dict:
        """每个在飞请求"已经处理到第几个 token"（= cache 的序列长度）。

        这是中断恢复所需要的进度读数，**引擎自己就能报**，不必去读 KV 张量：
        实测 38-token prefill 后 `get_seq_length() == 38`，每 decode 一步 +1。
        位置 = 该请求已经算过的 token 总数（prompt + 已生成）。

        注意：只有**活着的进程**能回答这个问题。设备被直接抱走/断电时问不到，
        所以控制面必须自己也能数（它收到了几个 token 它是知道的），
        引擎自报只作为校验与兜底。
        """
        positions = {}
        epoch = self._active_epoch or 0
        for (cached_epoch, request_id), cache in self._caches.items():
            if cached_epoch != epoch:
                continue
            try:
                positions[request_id] = int(cache.get_seq_length())
            except Exception:  # noqa: BLE001 - 不同版本接口不一
                continue
        return positions

    def status(self) -> dict:
        stage = self._active_stage()
        open_requests = sorted({k[1] for k in self._caches})
        return {
            **self.inventory(),
            "stage_calls": self.stage_calls,
            "open_requests": open_requests,
            "kv_bytes": {rid: self.kv_bytes(rid) for rid in open_requests},
            "request_positions": self.request_positions(),
            "device_allocated_bytes": self._device_allocated(),
            "layer_range": list(stage.layer_range) if stage else None,
        }

    # ------------------------------------------------------------- hidden 序列化

    def serialize_hidden(self, hidden: Any) -> tuple[bytes, dict]:
        """tensor → 原始字节（不经过 base64，避免 33% 膨胀）。"""
        tensor = hidden.detach().to("cpu")
        out_dtype = self.wire_dtype if self.wire_dtype is not None else tensor.dtype
        if out_dtype == torch.bfloat16:
            # numpy()/frombuffer() 不支持 bf16；链路统一落到 fp16（同为 2 字节，精度差异极小）
            out_dtype = torch.float16
        raw = tensor.contiguous().to(out_dtype).numpy().tobytes()
        meta = {
            "tensor_kind": "hf.hidden_states",
            "shape": list(tensor.shape),
            "dtype": str(out_dtype).replace("torch.", ""),
            "logical_bytes": len(raw),
        }
        return raw, meta

    def deserialize_hidden(self, payload: bytes, meta: Optional[dict] = None) -> tuple[Any, dict]:
        meta = meta or {"shape": [-1, self.hidden_size], "dtype": "float16"}
        dtype = getattr(torch, meta.get("dtype", "float16"))
        shape = tuple(int(x) for x in meta["shape"])
        expected = 1
        for dim in shape:
            expected *= dim
        if expected * torch.tensor([], dtype=dtype).element_size() != len(payload):
            raise ValueError(
                f"activation 字节数与 shape 不符: payload={len(payload)} shape={shape} dtype={dtype}"
            )
        flat = torch.frombuffer(bytearray(payload), dtype=dtype)
        hidden = flat.reshape(shape).to(device=self.device, dtype=self.dtype).contiguous()
        return hidden, meta

    # --------------------------------------------------------------- 内部实现

    def _forward(self, request_id: str, token_ids: Optional[list[int]] = None,
                 hidden: Any = None) -> StageForward:
        stage = self._require_active()
        start, end = stage.layer_range
        cache = self._cache_for(request_id)
        past = 0
        if cache.get_seq_length() > 0:
            past = cache.get_seq_length()

        if start == 0:
            # 索引张量必须与 embedding 同设备：GPU 上 index_select 不接受 CPU 索引
            ids = torch.tensor([token_ids or [0]], dtype=torch.long, device=self.device)
            hidden = stage.embed(ids).to(self.dtype)
        else:
            if hidden is None:
                raise ValueError("非首段前向必须携带上游 hidden states")
            hidden = hidden.to(self.dtype)

        seq = int(hidden.shape[1])
        # 位置/掩码同样必须落在计算设备上（RoPE 与 causal mask 会与 hidden 做运算）
        cache_position = torch.arange(past, past + seq, dtype=torch.long, device=self.device)
        position_ids = cache_position.unsqueeze(0)
        attention_mask = torch.ones(1, past + seq, dtype=torch.long, device=self.device)

        position_embeddings = self._position_embeddings(hidden, position_ids)
        mask = self._causal_mask(hidden, attention_mask, cache, position_ids)

        with torch.inference_mode():
            for layer in stage.layers:
                out = layer(
                    hidden_states=hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                hidden = out[0] if isinstance(out, (tuple, list)) else out

            token_id = None
            logits_last = None
            if stage.norm is not None and stage.lm_head is not None:
                normed = stage.norm(hidden[:, -1:, :])
                logits = stage.lm_head(normed)
                logits_last = logits[:, -1, :]
                token_id = int(torch.argmax(logits_last, dim=-1).item())

        self.stage_calls += 1
        metrics = {
            "node_id": self.node_id,
            "layer_range": [start, end],
            "layers": end - start,
            "device": str(self.device),
            "seq": seq,
            "past": past,
            "kv_bytes": self.kv_bytes(request_id),
            "hidden_bytes": int(hidden.numel() * hidden.element_size()),
            "is_final_stage": stage.norm is not None,
            "token_id": token_id,
        }
        keep_hidden = None if metrics["is_final_stage"] else hidden
        return StageForward(hidden=keep_hidden, token_id=token_id,
                            logits_last=logits_last, metrics=metrics)

    def _position_embeddings(self, hidden: Any, position_ids: Any) -> Any:
        rotary = getattr(self._rotary_holder, "rotary_emb", None)
        if rotary is None:
            return None
        return rotary(hidden, position_ids)

    def _causal_mask(self, hidden: Any, attention_mask: Any, cache: Any, position_ids: Any) -> Any:
        try:
            from transformers.masking_utils import create_causal_mask

            return create_causal_mask(
                self.config,
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                past_key_values=cache,
                position_ids=position_ids,
            )
        except Exception:  # pragma: no cover - 老版本 transformers 回退
            return None

    def _cache_for(self, request_id: str) -> Any:
        from transformers.cache_utils import DynamicCache

        key = (self._active_epoch or 0, request_id)
        cache = self._caches.get(key)
        if cache is None:
            cache = DynamicCache(config=self.config)
            self._caches[key] = cache
        return cache

    def _has_cache(self, request_id: str) -> bool:
        """本进程是否**已经有内容**的该请求 KV cache。

        token 级恢复能不能真的少算，取决于这一点：cache 不在（换了节点、
        引擎重启）时按最后一个 token 续算是错的。
        """
        cache = self._caches.get((self._active_epoch or 0, request_id))
        if cache is None:
            return False
        try:
            return int(cache.get_seq_length()) > 0
        except Exception:  # noqa: BLE001 - 不同 transformers 版本接口不一
            return True

    def _materialize_shard(self, start: int, end: int) -> tuple[Any, dict]:
        """建元骨架 + 只加载 [start,end) 的权重。"""
        from transformers import AutoModelForCausalLM

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(self.config)
        # 元设备上 RoPE 的 inv_freq 也是元张量，必须重建到真实设备；
        # 且要显式搬到 self.device——否则 GPU 上 position_embeddings 会留在 CPU。
        self._rotary_holder = model.model
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.__class__(self.config).to(self.device)

        state = self._shard_state_dict(start, end)
        model.load_state_dict(state, strict=False, assign=True)

        is_first = start == 0
        is_last = end == self.total_layers
        tie = bool(getattr(self.config, "tie_word_embeddings", False))
        if is_last and tie and not self.has_lm_head_weight:
            # 绑定权重：末段用 embed_tokens 当输出头（如实占用一份显存）
            model.lm_head.weight = model.model.embed_tokens.weight

        owned_tensors = [t for t in state.values() if torch.is_tensor(t)]
        nn = torch.nn
        layers = nn.ModuleList()
        for local_idx, global_idx in enumerate(range(start, end)):
            layer = model.model.layers[global_idx]
            layer.self_attn.layer_idx = local_idx
            layers.append(layer)

        embed = model.model.embed_tokens if (is_first or (is_last and tie)) else None
        norm = model.model.norm if is_last else None
        lm_head = model.lm_head if is_last else None

        for module in (layers, embed, norm, lm_head):
            if module is not None:
                module.to(self.device)
        if embed is not None:
            owned_tensors.append(embed.weight)

        return model, {
            "layers": layers,
            "embed": embed,
            "norm": norm,
            "lm_head": lm_head,
            "tensors": owned_tensors,
        }

    def _shard_state_dict(self, start: int, end: int) -> dict:
        from safetensors import safe_open

        wanted_prefixes = [f"model.layers.{i}." for i in range(start, end)]
        wanted_exact = set()
        if start == 0:
            wanted_exact.add("model.embed_tokens.weight")
        if end == self.total_layers:
            wanted_exact.add("model.norm.weight")
            if not self.has_lm_head_weight and getattr(self.config, "tie_word_embeddings", False):
                wanted_exact.add("model.embed_tokens.weight")
            else:
                wanted_exact.add("lm_head.weight")

        state: dict[str, Any] = {}
        for path in self.weight_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in wanted_exact or key.startswith(tuple(wanted_prefixes)):
                        tensor = handle.get_tensor(key)
                        # 统一到运行时 dtype：checkpoint 常是 bf16，而算子在
                        # CPU 上更适合 fp32；不转就会出现 Float/BFloat16 混算。
                        if tensor.is_floating_point() and tensor.dtype != self.dtype:
                            tensor = tensor.to(self.dtype)
                        state[key] = tensor
        return state

    @staticmethod
    def _weight_files(model_path: str) -> tuple[list[str], bool]:
        index = os.path.join(model_path, "model.safetensors.index.json")
        files: list[str] = []
        if os.path.exists(index):
            with open(index, "r", encoding="utf-8") as fh:
                weight_map = json.load(fh)["weight_map"]
            files = sorted({os.path.join(model_path, name) for name in weight_map.values()})
        else:
            single = os.path.join(model_path, "model.safetensors")
            if os.path.exists(single):
                files = [single]
            else:
                files = sorted(
                    os.path.join(model_path, name)
                    for name in os.listdir(model_path)
                    if name.endswith(".safetensors")
                )
        if not files:
            raise FileNotFoundError(f"{model_path} 下没有 safetensors 权重")
        has_lm_head = False
        from safetensors import safe_open

        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if any(key.startswith("lm_head.") for key in handle.keys()):
                    has_lm_head = True
                    break
        return files, has_lm_head

    def _resolve_device(self, device: str) -> Any:
        name = (device or "cpu").lower()
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(name)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"请求了 {name}，但本机 CUDA 不可用")
        return resolved

    def _resolve_dtype(self, dtype: str) -> Any:
        name = (dtype or "auto").lower()
        if name == "auto":
            return torch.float32 if self.device.type == "cpu" else torch.bfloat16
        mapping = {"float32": torch.float32, "fp32": torch.float32,
                   "float16": torch.float16, "fp16": torch.float16,
                   "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
        if name not in mapping:
            raise ValueError(f"未知 dtype {dtype}")
        return mapping[name]

    def _active_stage(self) -> Optional[_PreparedStage]:
        if self._active_epoch is None:
            return None
        return self._prepared.get(self._active_epoch)

    def _require_active(self) -> _PreparedStage:
        stage = self._active_stage()
        if stage is None:
            raise RuntimeError(f"[{self.node_id}] 没有活跃 epoch，先 prepare/activate")
        return stage

    def _device_total_bytes(self) -> int:
        if self.device.type == "cuda":
            try:
                return int(torch.cuda.get_device_properties(self.device).total_memory)
            except Exception:  # pragma: no cover
                return 0
        try:
            import psutil  # 可选

            return int(psutil.virtual_memory().total)
        except Exception:
            return 0

    def _device_allocated(self) -> int:
        if self.device.type == "cuda":
            try:
                return int(torch.cuda.memory_allocated(self.device))
            except Exception:  # pragma: no cover
                return 0
        return 0

    def _empty_device_cache(self) -> None:
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass

    @staticmethod
    def _token_ids(prompt: Any) -> list[int]:
        if prompt is None:
            return []
        if isinstance(prompt, str):
            return [int(b) for b in prompt.encode("utf-8")]
        return [int(t) for t in prompt]
