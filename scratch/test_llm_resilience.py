"""Verification script for LLM resilience and failover behavior."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.registry import LLMRegistry, RateLimitError, PROVIDER_MAP
from src.services.llm_scheduler import LLMScheduler
from src.services.reasoning_coordinator import ReasoningCoordinator


class TestZaiUrlResolution(unittest.TestCase):
    def test_zai_uses_coding_api_base_url(self):
        base_url, _ = PROVIDER_MAP["z_ai"]
        self.assertEqual(base_url, "https://api.z.ai/api/coding/paas/v4")


class TestPrimaryFallbackFailover(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_routes_to_fallback_on_primary_429(self):
        primary = MagicMock(spec=LLMRegistry)
        primary.is_degraded.return_value = True
        primary.provider = "openai"
        primary.chat_completion = AsyncMock(side_effect=RateLimitError("429"))

        fallback = MagicMock(spec=LLMRegistry)
        fallback.is_degraded.return_value = False
        fallback.provider = "nvidia"
        fallback.chat_completion = AsyncMock(return_value='{"action":"ABSTAIN"}')

        scheduler = LLMScheduler(
            registry=primary,
            model="gpt-4o-mini",
            fallback_registry=fallback,
            fallback_model="meta/llama-3.1-8b-instruct",
        )

        self.assertFalse(scheduler.is_degraded())

        task = asyncio.create_task(scheduler.process_queue())
        try:
            result = await scheduler.request_completion("system", "user")
            self.assertEqual(result, '{"action":"ABSTAIN"}')
            fallback.chat_completion.assert_awaited_once()
            primary.chat_completion.assert_not_awaited()
        finally:
            await scheduler.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestDouble429DegradedMode(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_degraded_when_both_rate_limited(self):
        primary = MagicMock(spec=LLMRegistry)
        primary.is_degraded.return_value = True

        fallback = MagicMock(spec=LLMRegistry)
        fallback.is_degraded.return_value = True

        scheduler = LLMScheduler(
            registry=primary,
            model="gpt-4o-mini",
            fallback_registry=fallback,
            fallback_model="meta/llama-3.1-8b-instruct",
        )

        self.assertTrue(scheduler.is_degraded())

        coordinator = ReasoningCoordinator(llm_scheduler=scheduler)
        from src.core.models import CandidateTrade

        candidate = CandidateTrade(
            symbol="BTCUSDT",
            anchor_symbol="ETHUSDT",
            proposed_side="BUY",
        )
        from src.models.reasoning import MarketContext, PortfolioSnapshot
        from src.intelligence.models import PromptContext

        decision = await coordinator.evaluate_candidate(
            candidate=candidate,
            market=MarketContext(symbol="BTCUSDT", timeframe="1m", current_price=50000.0),
            portfolio=PortfolioSnapshot(),
            evidence=PromptContext(),
        )
        self.assertEqual(decision.action, "ABSTAIN")
        self.assertEqual(decision.rationale, "LLM_DEGRADED_MODE")


class TestCooldownExpiration(unittest.TestCase):
    def test_degraded_mode_resets_after_cooldown(self):
        registry = LLMRegistry.__new__(LLMRegistry)
        registry._llm_degraded = True
        registry._cooldown_until = time.time() - 1
        self.assertFalse(registry.is_degraded())


class TestFallbackKeyManagement(unittest.TestCase):
    def test_get_api_keys_loads_fallback_llm_key(self):
        import src.main as main_module

        with patch.object(main_module, "keyring") as mock_keyring, patch.object(
            main_module, "os"
        ) as mock_os:
            mock_os.path.exists.return_value = False
            mock_keyring.get_password.side_effect = lambda service, key: {
                ("apex", "binance_key"): "bk",
                ("apex", "binance_secret"): "bs",
                ("apex", "llm_key"): "lk",
                ("apex", "fallback_llm_key"): "flk",
            }.get((service, key))

            keys = main_module.get_api_keys([])
            self.assertEqual(keys, ("bk", "bs", "lk", "flk"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
