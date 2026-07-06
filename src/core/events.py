from __future__ import annotations

import asyncio
import structlog
from collections import defaultdict
from typing import Awaitable, Callable

from src.core.models import SystemEvent

logger = structlog.get_logger("event_bus")

EXECUTION_EVENT_TYPES = frozenset({"EXECUTE_TRADE", "ORDER_FILLED"})


class EventBus:
    def __init__(self):
        self._execution_queue: asyncio.Queue[SystemEvent] = asyncio.Queue()
        self._general_queue: asyncio.Queue[SystemEvent] = asyncio.Queue()
        self._subscribers: dict[str, list[Callable[[SystemEvent], Awaitable[None]]]] = defaultdict(list)
        self._running = False
        self._dispatcher_tasks: list[asyncio.Task] = []

    async def publish(self, event: SystemEvent) -> None:
        if event.event_type in EXECUTION_EVENT_TYPES:
            queue = self._execution_queue
        else:
            queue = self._general_queue
        await queue.put(event)

    def publish_nowait(self, event: SystemEvent) -> None:
        if event.event_type in EXECUTION_EVENT_TYPES:
            queue = self._execution_queue
        else:
            queue = self._general_queue
        queue.put_nowait(event)
        logger.info(
            "EVENT_ENQUEUED",
            event_type=event.event_type,
            queue_size=self._execution_queue.qsize() + self._general_queue.qsize(),
        )
        logger.info(
            "EVENT_ENQUEUED",
            event_type=event.event_type,
            queue_size=self._execution_queue.qsize() + self._general_queue.qsize(),
        )

    def subscribe(self, event_type: str, callback: Callable[[SystemEvent], Awaitable[None]]) -> None:
        self._subscribers[event_type].append(callback)

    async def start_dispatcher(self) -> None:
        if self._running:
            return
        self._running = True
        self._dispatcher_tasks = [
            asyncio.create_task(self._run("execution", self._execution_queue)),
            asyncio.create_task(self._run("general", self._general_queue)),
        ]
        logger.info("EventBus dispatchers started", count=2)
        try:
            await asyncio.gather(*self._dispatcher_tasks)
        except asyncio.CancelledError:
            for t in self._dispatcher_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*self._dispatcher_tasks, return_exceptions=True)
            raise

    async def _run(self, name: str, queue: asyncio.Queue[SystemEvent]) -> None:
        logger.info("EventBus dispatcher started", queue=name)
        while self._running:
            try:
                event = await queue.get()
                logger.info(
                    "EVENT_DEQUEUED",
                    event_type=event.event_type,
                    queue_name=name,
                    queue_size=queue.qsize(),
                )
                callbacks = self._subscribers.get(event.event_type, []) + self._subscribers.get("*", [])
                for cb in callbacks:
                    cb_name = cb.__name__
                    logger.info(
                        "CALLBACK_START",
                        event_type=event.event_type,
                        callback=cb_name,
                        queue_name=name,
                    )
                    try:
                        await cb(event)
                    except Exception as e:
                        logger.error(
                            "Callback error in EventBus dispatcher",
                            event_type=event.event_type,
                            callback=cb_name,
                            queue_name=name,
                            error=str(e),
                        )
                    finally:
                        logger.info(
                            "CALLBACK_END",
                            event_type=event.event_type,
                            callback=cb_name,
                            queue_name=name,
                        )
            except Exception as e:
                logger.error("EventBus dispatcher error", queue_name=name, error=str(e))
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._running = False
