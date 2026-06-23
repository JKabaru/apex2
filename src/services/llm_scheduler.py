from __future__ import annotations

import asyncio
import time
import structlog

from src.llm.registry import LLMRegistry

logger = structlog.get_logger("llm_scheduler")

MIN_INTERVAL = 1.5
MAX_BACKOFF = 16.0


class LLMScheduler:
    def __init__(self, registry: LLMRegistry, model: str, audit_logger=None):
        self._registry = registry
        self._model = model
        self._audit_logger = audit_logger
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._last_request_time = 0.0
        self._running = False
        self._processor_task: asyncio.Task | None = None

    async def request_completion(self, system_prompt: str, user_prompt: str, priority: int = 0) -> str:
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        await self._queue.put({
            "future": future,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "priority": priority,
        })
        return await future

    async def process_queue(self) -> None:
        self._running = True
        logger.info("LLMScheduler processor started")
        while self._running:
            try:
                item = await self._queue.get()
                now = time.monotonic()
                elapsed = now - self._last_request_time
                if elapsed < MIN_INTERVAL:
                    await asyncio.sleep(MIN_INTERVAL - elapsed)

                backoff = 1.0
                while True:
                    try:
                        start = time.monotonic()
                        messages = [
                            {"role": "system", "content": item["system_prompt"]},
                            {"role": "user", "content": item["user_prompt"]},
                        ]
                        result = await self._registry.chat_completion(self._model, messages)
                        latency = time.monotonic() - start
                        self._last_request_time = time.monotonic()

                        self._log_usage(item["system_prompt"], item["user_prompt"], result, latency)
                        item["future"].set_result(result)
                        break
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "timeout" in error_str.lower():
                            logger.warning("LLMScheduler backoff", wait=backoff, error=error_str)
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, MAX_BACKOFF)
                        else:
                            logger.error("LLMScheduler request failed", error=error_str)
                            item["future"].set_exception(e)
                            break
            except Exception as e:
                logger.error("LLMScheduler processor error", error=str(e))
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._running = False

    def _log_usage(self, system_prompt: str, user_prompt: str, result: str, latency: float) -> None:
        if self._audit_logger is None:
            return
        token_estimate = len(system_prompt + user_prompt) // 4
        token_estimate_result = len(result) // 4
        logger.info(
            "LLM completion",
            input_tokens_est=token_estimate,
            output_tokens_est=token_estimate_result,
            latency_ms=round(latency * 1000),
        )
