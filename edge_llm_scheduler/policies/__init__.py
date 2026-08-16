"""edge_llm_scheduler.policies — 策略层。

机制 vs 策略分离：策略是可插拔的算法，决定"怎么做"；机制是固定流程，决定"做什么"。
当前提供接口 + 默认实现；复杂算法（按能力分任务/E2路由/优先级迁移/重并行化/token级恢复）
在 TODO 标记处后续补充。
"""

from .placement import (
    PlacementPolicy, DefaultPlacement, E2Placement, CapabilityPlacement,
)
from .migration import MigrationPolicy, DefaultMigration, PriorityMigration, MigrationPlan
from .reparallelization import (
    ReparallelizationPolicy, DefaultReparallelization, CapabilityReparallelization,
)
from .recovery import RecoveryPolicy, DefaultRecovery, TokenRecovery

__all__ = [
    "PlacementPolicy", "DefaultPlacement", "E2Placement", "CapabilityPlacement",
    "MigrationPolicy", "DefaultMigration", "PriorityMigration", "MigrationPlan",
    "ReparallelizationPolicy", "DefaultReparallelization", "CapabilityReparallelization",
    "RecoveryPolicy", "DefaultRecovery", "TokenRecovery",
]
