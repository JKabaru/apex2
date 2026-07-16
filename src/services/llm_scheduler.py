from __future__ import annotations

import asyncio
import time
import structlog

from src.llm.registry import LLMRegistry, RateLimitError, _is_rate_limit_error
from src.utils.token_bucket import TokenBucket

logger = structlog.get_logger("llm_scheduler")

MIN_INTERVAL = 1.5
MAX_BACKOFF = 16.0
QUEUE_TTL = 180.0


class LLMScheduler:
    def __init__(
        self,
        registry: LLMRegistry,
        model: str,
        audit_logger=None,
        fallback_registry: LLMRegistry | None = None,
        fallback_model: str | None = None,
        worker_count: int = 3,
    ):
        self._registry = registry
        self._model = model
        self._fallback_registry = fallback_registry
        self._fallback_model = fallback_model or ""
        self._audit_logger = audit_logger
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._last_request_time = 0.0
        self._running = False
        self._processor_tasks: list[asyncio.Task] = []
        self._worker_count = worker_count
        self._bucket = TokenBucket(capacity=5, refill_rate=1.0)
        self._llm_call_lock = asyncio.Lock()

    def is_degraded(self) -> bool:
        primary_degraded = self._registry.is_degraded()
        if not primary_degraded:
            return False
        if (
            self._fallback_registry is not None
            and self._fallback_model
            and not self._fallback_registry.is_degraded()
        ):
            return False
        return True

    def _active_route(self) -> tuple[LLMRegistry, str, str]:
        if self._registry.is_degraded():
            if (
                self._fallback_registry is not None
                and self._fallback_model
                and not self._fallback_registry.is_degraded()
            ):
                return self._fallback_registry, self._fallback_model, "fallback"
        return self._registry, self._model, "primary"

    async def request_completion(self, system_prompt: str, user_prompt: str, priority: int = 0) -> str:
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        await self._queue.put({
            "future": future,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "priority": priority,
            "enqueued_at": time.monotonic(),
        })
        try:
            return await asyncio.wait_for(future, timeout=120.0)
        except asyncio.TimeoutError:
            raise TimeoutError("LLM request timed out after 120s in scheduler queue")

    async def start(self) -> None:
        self._running = True
        self._processor_tasks = [
            asyncio.create_task(self._worker())
            for _ in range(self._worker_count)
        ]
        logger.info("LLMScheduler workers started", count=self._worker_count)

    async def _worker(self) -> None:
        while self._running:
            try:
                item = await self._queue.get()

                # Discard stale requests that have been queued too long
                age = time.monotonic() - item["enqueued_at"]
                if age > QUEUE_TTL:
                    logger.warning(
                        "Discarding stale LLM request",
                        age_seconds=round(age, 1),
                    )
                    if not item["future"].done():
                        item["future"].set_exception(
                            TimeoutError(f"LLM request stale after {age:.1f}s in queue")
                        )
                    continue

                backoff = 1.0
                while True:
                    try:
                        async with self._llm_call_lock:
                            now = time.monotonic()
                            elapsed = now - self._last_request_time
                            if elapsed < MIN_INTERVAL:
                                await asyncio.sleep(MIN_INTERVAL - elapsed)

                            registry, model, route = self._active_route()
                            if route == "fallback":
                                logger.info(
                                    "Routing LLM request to fallback provider",
                                    provider=registry.provider,
                                    model=model,
                                )

                            start = time.monotonic()
                            messages = [
                                {"role": "system", "content": item["system_prompt"]},
                                {"role": "user", "content": item["user_prompt"]},
                            ]
                            await self._bucket.acquire()
                            result = await registry.chat_completion(model, messages)
                            latency = time.monotonic() - start
                            self._last_request_time = time.monotonic()

                        self._log_usage(item["system_prompt"], item["user_prompt"], result, latency)
                        if not item["future"].done():
                            item["future"].set_result(result)
                        else:
                            logger.warning("LLMScheduler future already done (caller timed out)")
                        break
                    except RateLimitError as e:
                        if not item["future"].done():
                            item["future"].set_exception(e)
                        else:
                            logger.warning("LLMScheduler future already done (caller timed out)")
                        break
                    except Exception as e:
                        error_str = str(e)
                        if _is_rate_limit_error(e) or "timeout" in error_str.lower() or "empty" in error_str.lower():
                            logger.warning("LLMScheduler backoff", wait=backoff, error=error_str)
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, MAX_BACKOFF)
                        else:
                            logger.error("LLMScheduler request failed", error=error_str)
                            if not item["future"].done():
                                item["future"].set_exception(e)
                            else:
                                logger.warning("LLMScheduler future already done (caller timed out)")
                            break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("LLMScheduler worker error", error=str(e))
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._running = False
        for t in self._processor_tasks:
            if not t.done():
                t.cancel()
        if self._processor_tasks:
            await asyncio.gather(*self._processor_tasks, return_exceptions=True)
            self._processor_tasks.clear()

    def _log_usage(self, system_prompt: str, user_prompt: str, result: str, latency: float) -> None:
        if self._audit_logger is None:
            return
        token_estimate = len(system_prompt + user_prompt) // 4
        token_estimate_result = len(result) // 4
        preview = result[:500] if result else ""
        logger.info(
            "LLM completion",
            input_tokens_est=token_estimate,
            output_tokens_est=token_estimate_result,
            latency_ms=round(latency * 1000),
            response_preview=preview,
        )
