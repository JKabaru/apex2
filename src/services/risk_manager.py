from __future__ import annotations

import structlog

from src.api.binance_client import BinanceClient
from src.core.events import EventBus
from src.core.models import CandidateTrade, RiskDecision
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("risk_manager")

MAX_CONCURRENT_POSITIONS = 3
MIN_LLM_CONFIDENCE = 0.3
MAX_LIVE_EXPOSURE_USDT = 10000.0


class RiskManager:
    def __init__(
        self,
        client: BinanceClient,
        event_bus: EventBus,
        max_positions: int = MAX_CONCURRENT_POSITIONS,
        min_llm_confidence: float = MIN_LLM_CONFIDENCE,
        max_live_exposure_usdt: float = MAX_LIVE_EXPOSURE_USDT,
        take_profit_pct: float = 1.04,
    ):
        self._client = client
        self._event_bus = event_bus
        self._max_positions = max_positions
        self._min_llm_confidence = min_llm_confidence
        self._max_live_exposure_usdt = max_live_exposure_usdt
        self._take_profit_pct = take_profit_pct

    async def evaluate_candidate(
        self, candidate: CandidateTrade, portfolio: PortfolioManager, llm_confidence: float = 0.0
    ) -> tuple[RiskDecision, str]:
        logger.info(
            "RiskManager.evaluate_candidate START",
            symbol=candidate.symbol,
            llm_confidence=llm_confidence,
            min_llm_confidence=self._min_llm_confidence,
            max_positions=self._max_positions,
        )
        # ── Exchange-first validation (Exchange is God) ──
        try:
            exchange_positions = await self._client.get_open_positions()
        except Exception as e:
            logger.critical(
                "Exchange position query failed — rejecting trade",
                error=str(e),
            )
            return RiskDecision.REJECTED_CONSTRAINT, "EXCHANGE_QUERY_FAILED"

        exchange_open_count = len(exchange_positions)

        # Desync detection: exchange has positions missing from local state
        local_open_count = len(portfolio.get_live_open_positions())
        if exchange_open_count > local_open_count:
            logger.warning(
                "Mid-run desync detected. Exchange has positions missing locally. "
                "Triggering immediate reconciliation to ADOPT positions.",
                exchange_count=exchange_open_count,
                local_count=local_open_count,
            )
            from src.services.reconciler import Reconciler
            try:
                await Reconciler.sync_missing_positions_from_exchange(self._client, portfolio, take_profit_pct=self._take_profit_pct)
            except Exception as e:
                logger.error("Adoption sync failed — will retry next cycle", error=str(e))
            return RiskDecision.REJECTED_CONSTRAINT, "EXCHANGE_DESYNC_SYNCING"

        # Exchange-level position cap (authoritative)
        if exchange_open_count >= self._max_positions:
            logger.warning(
                "Risk rejected: exchange concurrent position limit reached",
                exchange_count=exchange_open_count,
                max=self._max_positions,
                symbol=candidate.symbol,
            )
            return RiskDecision.REJECTED_CONSTRAINT, "EXCHANGE_LIMIT_REACHED"

        if llm_confidence < self._min_llm_confidence:
            logger.warning(
                "Risk rejected: low LLM confidence",
                symbol=candidate.symbol,
                confidence=llm_confidence,
                threshold=self._min_llm_confidence,
            )
            return RiskDecision.REJECTED_QUALITY, "LOW_CONFIDENCE"

        open_positions = portfolio.get_live_open_positions()
        if len(open_positions) >= self._max_positions:
            logger.warning(
                "Risk rejected: max concurrent positions reached",
                current=len(open_positions),
                max=self._max_positions,
                symbol=candidate.symbol,
            )
            return RiskDecision.REJECTED_CONSTRAINT, "MAX_POSITIONS"

        exposure = portfolio.get_live_exposure(candidate.symbol)
        if exposure > 0:
            logger.warning(
                "Risk rejected: duplicate symbol exposure",
                symbol=candidate.symbol,
                exposure=exposure,
            )
            return RiskDecision.REJECTED_CONSTRAINT, "DUPLICATE_SYMBOL"

        total_live_exposure = portfolio.get_total_live_exposure()
        if total_live_exposure >= self._max_live_exposure_usdt:
            logger.warning(
                "Risk rejected: max live exposure reached",
                current=total_live_exposure,
                max=self._max_live_exposure_usdt,
                symbol=candidate.symbol,
            )
            return RiskDecision.REJECTED_CONSTRAINT, "MAX_EXPOSURE"

        return RiskDecision.APPROVED, ""
