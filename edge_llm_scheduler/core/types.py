"""核心数据类型定义。

所有模块共享的类型：节点、模型、请求、任务、KV 块、事件、结果。
这是框架的地基，其它模块都依赖本文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ============================================================
# 节点（Node）
# ============================================================

class NodeRole(Enum):
    """节点可承担的角色。边缘异构系统里节点不一定全角色。"""
    PREFILL = auto()      # 专职 prefill
    DECODE = auto()       # 专职 decode
    CACHE = auto()        # 专职 KV 缓存
    EXPERT = auto()       # 专职专家计算
    GENERAL = auto()      # 通用（全角色）


@dataclass
class NodeCapability:
    """节点的静态能力（profiling 一次，之后不变）。"""
    compute_flops: float = 0.0          # 算力（TFLOPS）
    memory_total: int = 0               # 总显存（字节）
    bandwidth: float = 0.0              # 链路带宽（MB/s，相对调度器的）
    supported_roles: set = field(default_factory=lambda: {NodeRole.GENERAL})
    latency_ms: float = 0.0             # 链路延迟（毫秒）


@dataclass
class NodeState:
    """节点的动态状态（心跳/探测实时更新）。"""
    memory_free: int = 0                # 剩余显存（字节）
    load: float = 0.0                   # 当前负载 [0, 1]
    alive: bool = True                  # 是否存活
    kv_blocks: int = 0                  # 已持有 KV 块数
    expert_ids: set = field(default_factory=set)   # 已装载的专家 id（GPU 内）
    expert_ids_cpu: set = field(default_factory=set)  # CPU/下级存储的专家 id
    layer_range: Optional[tuple] = None # 已装载的层区间 (start, end) 含 start 不含 end
    queue_depth: int = 0                # 等待队列长度
    pending_tasks: set = field(default_factory=set)  # 正在执行的 task_id


@dataclass
class Node:
    """一个算力单元（GPU/手机/边缘盒）。"""
    node_id: str
    capability: NodeCapability
    state: NodeState = field(default_factory=NodeState)

    def __repr__(self) -> str:
        return f"Node({self.node_id}, mem_free={self.state.memory_free}, load={self.state.load:.2f})"


# ============================================================
# 模型（Model）
# ============================================================

@dataclass
class ExpertSpec:
    """一个专家（MoE 中的 FFN 子网络）的静态描述。"""
    expert_id: int
    layer_id: int                # 所属层
    size_bytes: int              # 参数量字节


@dataclass
class ModelSpec:
    """模型静态结构。"""
    model_name: str
    num_layers: int
    hidden_size: int
    num_experts_per_layer: int = 0       # 0 = dense（无 MoE）
    experts: list = field(default_factory=list)  # list[ExpertSpec]
    total_params_bytes: int = 0
    kv_bytes_per_token: int = 0          # 每 token 的 KV 占用（字节）
    # 注意力权重（每设备可全复制，因为小）——按模型算
    attention_params_bytes: int = 0

    def is_moe(self) -> bool:
        return self.num_experts_per_layer > 0


@dataclass
class ModelPlacement:
    """模型装载决策：某节点装哪些层/哪些专家/专家进哪级存储。"""
    node_id: str
    layer_range: Optional[tuple] = None   # 持有的层 (start, end)
    experts_gpu: list = field(default_factory=list)   # GPU 内专家 id
    experts_cpu: list = field(default_factory=list)   # CPU/下级存储专家 id

    def total_expert_count(self) -> int:
        return len(self.experts_gpu) + len(self.experts_cpu)


# ============================================================
# KV 块（KVBlock）
# ============================================================

@dataclass
class KVBlock:
    """KV 缓存的最小管理单位（对齐 vLLM/PagedAttention 的分页思想）。"""
    block_hash: int              # 内容哈希（前缀标识）
    num_tokens: int              # 本块覆盖的 token 数
    byte_size: int               # 大小（字节）
    layer_range: Optional[tuple] = None  # 覆盖的层区间（用于部分迁移）
    data: Any = None             # 实际数据（mock 用 bytes；真实走 LMCache）

    # 迁移优先级辅助（参考 Preble 的 PT×N 公式）
    reuse_count: int = 0         # 历史被共享的请求数（N_j）
    prefill_time_ms: float = 0.0 # 重算这段 KV 的 prefill 耗时（PT_j）
    is_hot: bool = False         # 是否热块（Mooncake 式标记）

    @property
    def priority(self) -> float:
        """迁移价值 = 复用度 × 重算成本（Preble 的 M_i = PT×N）。"""
        return self.reuse_count * self.prefill_time_ms


# ============================================================
# 请求与任务（Request / Task）
# ============================================================

@dataclass
class InferenceRequest:
    """一次推理请求。"""
    request_id: str
    prompt: Any = None           # str 或 token 列表
    max_tokens: int = 64
    arrival_time: float = 0.0
    user_id: Optional[str] = None


@dataclass
class Task:
    """下发到某节点的子任务（一个请求可能拆成多个 Task）。"""
    task_id: str
    request_id: str
    node_id: str
    layer_range: Optional[tuple] = None
    kv_block_ids: list = field(default_factory=list)
    status: str = "pending"      # pending/running/done/failed/interrupted
    create_time: float = 0.0
    finish_time: Optional[float] = None
    result: Any = None

    def mark(self, status: str) -> None:
        self.status = status


@dataclass
class GenerationResult:
    """推理结果。"""
    request_id: str
    text: str = ""
    num_tokens: int = 0
    kv_block_ids: list = field(default_factory=list)  # 新产出的 KV 块
    latency_ms: float = 0.0
    hit_tokens: int = 0          # 缓存命中 token 数
    error: Optional[str] = None


# ============================================================
# 事件（Event）
# ============================================================

class EventType(Enum):
    REQUEST_ARRIVED = auto()
    NODE_JOINED = auto()
    NODE_LEFT = auto()
    HEARTBEAT = auto()
    TASK_DONE = auto()
    TASK_INTERRUPTED = auto()
    KV_MIGRATED = auto()
    KV_MISS = auto()             # KV 未命中（可触发迁移/预取）


@dataclass
class Event:
    type: EventType
    payload: Any = None
    timestamp: float = 0.0
    node_id: Optional[str] = None
    request_id: Optional[str] = None
