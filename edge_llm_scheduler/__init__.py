"""edge_llm_scheduler — 分布式算力调度框架。

架构分层：
- core/     机制层（类型、KV存储抽象、传输抽象、事件总线、节点管理、模型装载、任务调度）
- policies/ 策略层（可插拔算法：放置/迁移/重并行化/中断恢复）
- backends/ 对接实现（mock / vLLM / LMCache / TCP / WiFi）
"""

from .config import SchedulerConfig

__version__ = "0.1.0"
__all__ = ["SchedulerConfig", "__version__"]
