"""edge_llm_scheduler.core — 框架机制层。

包含：核心类型、KV 存储抽象、传输抽象、事件总线、节点管理、模型装载、任务调度、请求流。
"""

from .types import (
    Node, NodeCapability, NodeState, NodeRole,
    ActivationEnvelope, ModelSpec, ExpertSpec, ModelPlacement, PipelinePlan, StageAssignment,
    KVBlock, InferenceRequest, Task, GenerationResult,
    Event, EventType,
)
from .storage import KVStore
from .transport import Transport
from .event_bus import EventBus
from .node_manager import NodeManager
from .model_manager import ModelManager
from .task_scheduler import TaskScheduler, Engine
from .request_flow import RequestFlow
from .stage_runtime import StagePrepareResult, StageRuntime
from .pipeline_controller import (
    PipelineReconfigurationCoordinator,
    ReconfigurationReport,
    StateTransferRecord,
)

__all__ = [
    "Node", "NodeCapability", "NodeState", "NodeRole",
    "ActivationEnvelope", "ModelSpec", "ExpertSpec", "ModelPlacement", "PipelinePlan", "StageAssignment",
    "KVBlock", "InferenceRequest", "Task", "GenerationResult",
    "Event", "EventType",
    "KVStore", "Transport", "EventBus", "NodeManager",
    "ModelManager", "TaskScheduler", "Engine", "RequestFlow", "StageRuntime", "StagePrepareResult",
    "PipelineReconfigurationCoordinator", "ReconfigurationReport", "StateTransferRecord",
]
