import asyncio
import time


class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._refill_rate
                self._tokens = 0.0
                self._last_refill = now + wait
            else:
                self._tokens -= 1.0
                return

        await asyncio.sleep(wait)

        async with self._lock:
            self._tokens -= 1.0
