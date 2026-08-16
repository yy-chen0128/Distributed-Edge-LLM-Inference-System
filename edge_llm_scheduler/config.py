"""配置加载：dataclass + 从 dict/YAML/env 读。

最小配置即可启动（都用默认值）；后续扩展各策略参数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SchedulerConfig:
    # 通用
    model_name: str = "mock-model"
    num_layers: int = 32
    hidden_size: int = 4096
    num_experts_per_layer: int = 0          # 0 = dense

    # 心跳/超时
    heartbeat_timeout_sec: float = 10.0
    heartbeat_interval_sec: float = 3.0

    # 传输
    transport_backend: str = "mock"          # mock / tcp / wifi
    bandwidth_mbps: float = 100.0
    latency_ms: float = 10.0

    # 存储
    storage_backend: str = "mock"            # mock / lmcache

    # 引擎
    engine_backend: str = "mock"             # mock / vllm
    tokens_per_sec: float = 20.0

    # 策略（默认实现）
    placement_policy: str = "default"
    migration_policy: str = "default"
    reparallelization_policy: str = "default"
    recovery_policy: str = "default"

    @classmethod
    def from_dict(cls, d: dict) -> "SchedulerConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        """从环境变量读（LMCACHE_ 风格前缀 ELS_）。"""
        def env(name: str, default):
            return os.environ.get(f"ELS_{name.upper()}", default)
        return cls(
            model_name=env("model_name", "mock-model"),
            num_layers=int(env("num_layers", 32)),
            transport_backend=env("transport_backend", "mock"),
            storage_backend=env("storage_backend", "mock"),
            engine_backend=env("engine_backend", "mock"),
            bandwidth_mbps=float(env("bandwidth_mbps", 100)),
        )
