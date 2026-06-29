from __future__ import annotations

import json
import uuid
import structlog

from src.core.models import CandidateTrade, SystemEvent
from src.core.events import EventBus
from src.services.llm_scheduler import LLMScheduler
from src.services.market_context import MarketContextService

logger = structlog.get_logger("scanner")

MIN_CORRELATION = 0.6
MAX_P_VALUE = 0.05
SCAN_INTERVAL = 30


SYSTEM_PROMPT = """You are a quantitative trading analyst. Given market data for a symbol, respond with a JSON object containing:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <0.0-1.0>,
  "rationale": "<brief explanation>"
}
Only respond with the JSON object, no other text."""


class MarketScanner:
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus

    async def run_scan_cycle(
        self,
        alternates: list[str],
        context: MarketContextService,
        llm: LLMScheduler,
    ) -> None:
        for symbol in alternates:
            try:
                candidate = CandidateTrade(
                    symbol=symbol,
                    anchor_symbol="",
                    proposed_side="BUY",
                )

                state = await context.get_state(symbol)

                if state["current_price"] == 0.0:
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

                if correlations:
                    best = correlations[0]
                    candidate.correlation_score = abs(best.get("coefficient", 0))
                    candidate.anchor_symbol = best.get("anchor", "")
                    candidate.proposed_side = "BUY" if best.get("direction", 0) >= 0 else "SELL"

                user_prompt = (
                    f"Symbol: {symbol}\n"
                    f"Current Price: {state['current_price']}\n"
                    f"Indicators: {json.dumps(state['indicators'])}\n"
                    f"Top Correlations: {json.dumps(correlations, default=str)}\n\n"
                    f"Should we BUY, SELL, or HOLD?"
                )

                try:
                    response = await llm.request_completion(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                    )
                except Exception as e:
                    logger.error("LLM request failed in scanner", symbol=symbol, error=str(e))
                    continue

                if not response or not response.strip():
                    logger.warning("LLM returned empty response, skipping", symbol=symbol)
                    continue

                try:
                    decision = json.loads(response.strip())
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse LLM response", symbol=symbol, error=str(e), raw_preview=response[:100])
                    continue

                action = decision.get("action", "HOLD")
                if action not in ("BUY", "SELL"):
                    logger.debug("LLM decided HOLD", symbol=symbol, rationale=decision.get("rationale", ""))
                    continue

                candidate.proposed_side = action
                candidate.signal_strength = decision.get("confidence", 0.5)

                correlation_id = str(uuid.uuid4())
                event = SystemEvent(
                    event_type="CANDIDATE_DISCOVERED",
                    service_name="Scanner",
                    payload={
                        "candidate": candidate.model_dump(),
                        "llm_confidence": decision.get("confidence", 0.5),
                        "llm_rationale": decision.get("rationale", ""),
                        "llm_request_id": None,
                        "correlation_id": correlation_id,
                        "strategy_version": "1.0",
                    },
                )
                await self._event_bus.publish(event)
                logger.info(
                    "Candidate discovered",
                    symbol=symbol,
                    action=action,
                    confidence=decision.get("confidence"),
                    correlation_id=correlation_id,
                )

            except Exception as e:
                logger.error("Scan cycle error for symbol", symbol=symbol, error=str(e))
