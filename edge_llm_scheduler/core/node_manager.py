"""节点生命周期管理：注册、心跳、状态维护、加入/离开事件发布。

节点加入 → 评估能力 → 发布 NODE_JOINED（触发放置策略）
节点离开 → 发布 NODE_LEFT（触发迁移/重并行化策略）
心跳   → 更新动态状态（内存/负载/存活）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from .event_bus import EventBus
from .types import Event, EventType, Node, NodeState

logger = logging.getLogger(__name__)

# 心跳超时判定阈值（秒）
HEARTBEAT_TIMEOUT_SEC = 10.0


class NodeManager:
    def __init__(self, event_bus: Optional[EventBus] = None, heartbeat_timeout_sec: float = HEARTBEAT_TIMEOUT_SEC) -> None:
        self._nodes: Dict[str, Node] = {}
        self._last_heartbeat: Dict[str, float] = {}
        self._event_bus = event_bus
        self.heartbeat_timeout_sec = heartbeat_timeout_sec

    # ---------- 查询 ----------

    def get(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list:
        return list(self._nodes.values())

    def alive_nodes(self) -> list:
        return [n for n in self._nodes.values() if n.state.alive]

    def healthy_nodes(self) -> list:
        """存活且负载未满的节点。"""
        return [n for n in self.alive_nodes() if n.state.load < 1.0]

    def count(self) -> int:
        return len(self._nodes)

    # ---------- 生命周期 ----------

    async def register(self, node: Node) -> None:
        """节点加入。发布 NODE_JOINED 事件。"""
        self._nodes[node.node_id] = node
        self._last_heartbeat[node.node_id] = time.time()
        logger.info(f"node registered: {node}")
        if self._event_bus:
            await self._event_bus.publish(
                Event(type=EventType.NODE_JOINED, payload=node, node_id=node.node_id, timestamp=time.time())
            )

    async def unregister(self, node_id: str, reason: str = "manual") -> None:
        """节点离开。标记不可用并发布 NODE_LEFT。"""
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.state.alive = False
        logger.warning(f"node leaving: {node_id} ({reason})")
        if self._event_bus:
            await self._event_bus.publish(
                Event(type=EventType.NODE_LEFT, payload=node, node_id=node_id, timestamp=time.time())
            )

    async def update_heartbeat(self, node_id: str, state: Optional[NodeState] = None) -> None:
        """心跳更新。state 非空则替换动态状态。"""
        node = self._nodes.get(node_id)
        if node is None:
            logger.warning(f"heartbeat from unknown node: {node_id}")
            return
        if state is not None:
            node.state = state
        node.state.alive = True
        self._last_heartbeat[node_id] = time.time()
        if self._event_bus:
            await self._event_bus.publish(
                Event(type=EventType.HEARTBEAT, node_id=node_id, timestamp=time.time(), payload=node.state)
            )

    async def check_stale_nodes(self) -> None:
        """周期检查：超时未心跳的节点标记为离开。"""
        now = time.time()
        stale = [
            nid for nid, t in self._last_heartbeat.items()
            if now - t > self.heartbeat_timeout_sec and self._nodes.get(nid, None) is not None
            and self._nodes[nid].state.alive
        ]
        for nid in stale:
            logger.warning(f"node stale (no heartbeat): {nid}")
            await self.unregister(nid, reason="heartbeat_timeout")

    async def start_stale_monitor(self, interval_sec: float = 3.0) -> None:
        """后台周期检查超时节点。"""
        while True:
            await self.check_stale_nodes()
            await asyncio.sleep(interval_sec)
