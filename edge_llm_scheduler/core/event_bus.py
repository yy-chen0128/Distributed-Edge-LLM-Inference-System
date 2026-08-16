"""异步事件总线。

节点加入/离开/请求到达/任务完成等异步事件通过 EventBus 分发。
机制：publish 进队列，一个分发协程按订阅者派发。订阅者按事件类型注册。

这是 Preble `runtime_request_queue + 后台协程` 模式的泛化。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Set

from .types import Event, EventType

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """注册某类事件的处理器。handler 签名：async def handler(event: Event) -> None"""
        self._subscribers.setdefault(event_type, []).append(handler)

    def subscribe_many(self, handlers: Dict[EventType, Callable]) -> None:
        for et, h in handlers.items():
            self.subscribe(et, h)

    async def publish(self, event: Event) -> None:
        """发布事件（非阻塞，入队即可）。"""
        await self._queue.put(event)

    async def run(self) -> None:
        """启动事件分发主循环。"""
        self._running = True
        logger.info("EventBus started")
        while self._running:
            event = await self._queue.get()
            await self._dispatch(event)
            self._queue.task_done()

    def start(self) -> None:
        """后台启动分发循环（asyncio 单进程模式）。"""
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _dispatch(self, event: Event) -> None:
        handlers = self._subscribers.get(event.type, [])
        if not handlers:
            logger.debug(f"no handler for {event.type}")
            return
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.exception(f"handler {handler} failed for {event.type}: {e}")
