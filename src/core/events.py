from __future__ import annotations

import asyncio
import structlog
from collections import defaultdict
from typing import Awaitable, Callable

from src.core.models import SystemEvent

logger = structlog.get_logger("event_bus")


class EventBus:
    def __init__(self):
        self._queue: asyncio.Queue[SystemEvent] = asyncio.Queue()
        self._subscribers: dict[str, list[Callable[[SystemEvent], Awaitable[None]]]] = defaultdict(list)
        self._running = False
        self._dispatcher_task: asyncio.Task | None = None

    async def publish(self, event: SystemEvent) -> None:
        await self._queue.put(event)

    def subscribe(self, event_type: str, callback: Callable[[SystemEvent], Awaitable[None]]) -> None:
        self._subscribers[event_type].append(callback)

    async def start_dispatcher(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info("EventBus dispatcher started")
        while self._running:
            try:
                event = await self._queue.get()
                callbacks = self._subscribers.get(event.event_type, []) + self._subscribers.get("*", [])
                for cb in callbacks:
                    try:
                        await cb(event)
                    except Exception as e:
                        logger.error(
                            "Callback error in EventBus dispatcher",
                            event_type=event.event_type,
                            callback=cb.__name__,
                            error=str(e),
                        )
            except Exception as e:
                logger.error("EventBus dispatcher error", error=str(e))
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._running = False
