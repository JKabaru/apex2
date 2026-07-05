from __future__ import annotations

import uuid
import structlog

from src.core.models import CandidateTrade, SystemEvent
from src.core.events import EventBus
from src.services.market_context import MarketContextService

logger = structlog.get_logger("scanner")

MIN_CORRELATION = 0.6
MAX_P_VALUE = 0.05
SCAN_INTERVAL = 30


class MarketScanner:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus

    async def run_scan_cycle(
        self,
        alternates: list[str],
        context: MarketContextService,
    ) -> None:
        for symbol in alternates:
            try:
                state = await context.get_state(symbol)

                if state.get("current_price", 0.0) == 0.0:
                    logger.warning("No price data available, skipping", symbol=symbol)
                    continue

                correlations = state.get("correlations", [])
                deterministic_pass = any(
                    abs(c.get("coefficient", 0)) >= MIN_CORRELATION and c.get("p_value", 1) <= MAX_P_VALUE
                    for c in correlations
                )
                if not deterministic_pass:
                    logger.debug("Deterministic gate rejected", symbol=symbol)
                    continue

                best = correlations[0]
                correlation_score = abs(best.get("coefficient", 0))
                anchor_symbol = best.get("anchor", "")
                proposed_side = "BUY" if best.get("direction", 0) >= 0 else "SELL"

                candidate = CandidateTrade(
                    symbol=symbol,
                    anchor_symbol=anchor_symbol,
                    correlation_score=correlation_score,
                    signal_strength=correlation_score,
                    proposed_side=proposed_side,
                )

                correlation_id = str(uuid.uuid4())
                opportunity_id = str(uuid.uuid4())
                candidate.opportunity_id = opportunity_id

                event = SystemEvent(
                    event_type="CANDIDATE_DISCOVERED",
                    service_name="Scanner",
                    payload={
                        "candidate": candidate.model_dump(),
                        "correlation_id": correlation_id,
                        "opportunity_id": opportunity_id,
                        "timeframe": "5m",
                        "strategy_version": "1.0",
                    },
                )
                await self._event_bus.publish(event)
                logger.info(
                    "Candidate discovered",
                    symbol=symbol,
                    action=proposed_side,
                    correlation_score=correlation_score,
                    correlation_id=correlation_id,
                )

            except Exception as e:
                logger.error("Scan cycle error for symbol", symbol=symbol, error=str(e))
