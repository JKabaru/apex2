from __future__ import annotations

import asyncio
import json
import structlog
from datetime import datetime, timezone

from src.core.events import EventBus
from src.core.models import Position, PositionState, SystemEvent
from src.services.execution import ExecutionService
from src.services.llm_scheduler import LLMScheduler
from src.services.market_context import MarketContextService
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("position_manager")

MONITOR_INTERVAL = 5
REVIEW_HOLDING_HOURS = 4
REVIEW_CORRELATION_THRESHOLD = 0.4

REVIEW_SYSTEM_PROMPT = """You are a risk reviewer. A position has exceeded its holding time or lost correlation strength. 
Respond with JSON: {"recommendation": "HOLD"|"CLOSE", "rationale": "..."}"""

UNMANAGED_ADOPTED_REVIEW_PROMPT = """WARNING: This is an UNMANAGED_ADOPTED position. You have no historical entry thesis or original correlation signal for this trade. You must evaluate this position purely as a brand-new entry candidate based on the current market context. If the current indicators and correlations do not strongly justify opening a NEW position right now, you MUST output CLOSE immediately to eliminate risk and free up capital. Do not default to HOLD just because you lack context.
Respond with JSON: {"recommendation": "HOLD"|"CLOSE", "rationale": "..."}"""


class PositionManager:
    def __init__(
        self,
        portfolio: PortfolioManager,
        execution_svc: ExecutionService,
        llm_scheduler: LLMScheduler,
        event_bus: EventBus,
    ):
        self._portfolio = portfolio
        self._execution = execution_svc
        self._llm = llm_scheduler
        self._event_bus = event_bus
        self._event_bus.subscribe("ORDER_FILLED", self._on_order_filled)
        logger.info("PositionManager initialized")

    async def _on_order_filled(self, event: SystemEvent) -> None:
        payload = event.payload
        fill_type = payload.get("type")

        if fill_type == "entry":
            try:
                position = Position(
                    symbol=payload["symbol"],
                    side=payload["side"],
                    quantity=payload["executed_qty"],
                    avg_fill_price=payload["avg_price"],
                    fees=payload.get("commission", 0.0),
                    exchange_order_ids=[str(payload.get("order_id", ""))],
                    anchor_symbol=payload.get("anchor_symbol", ""),
                    correlation_score=payload.get("correlation_score", 0.0),
                    initial_stop_loss=payload.get("initial_stop_loss", 0.0),
                    initial_take_profit=payload.get("initial_take_profit", 0.0),
                    current_stop=payload.get("initial_stop_loss", 0.0),
                    current_target=payload.get("initial_take_profit", 0.0),
                    entry_thesis=payload.get("entry_thesis", ""),
                    lifecycle_state=PositionState.OPEN,
                )
                await self._portfolio.add_position(position)
                logger.info(
                    "Position opened from fill",
                    position_id=position.position_id,
                    symbol=position.symbol,
                    side=position.side,
                )
            except Exception as e:
                logger.error("Failed to create position from entry fill", error=str(e))

        elif fill_type == "exit":
            try:
                position_id = payload["position_id"]
                reason = payload.get("reason", "manual")
                await self._portfolio.update_position_state(
                    position_id,
                    PositionState.CLOSED,
                    exit_reason=reason,
                )
                logger.info(
                    "Position closed from fill",
                    position_id=position_id,
                    reason=reason,
                )
            except Exception as e:
                logger.error("Failed to close position from exit fill", error=str(e))

    async def monitor_positions(self, context: MarketContextService) -> None:
        logger.info("Position monitor loop started")
        while True:
            try:
                await asyncio.sleep(MONITOR_INTERVAL)
                open_positions = self._portfolio.get_open_positions()

                for pos in open_positions:
                    try:
                        state = await context.get_state(pos.symbol)
                        current_price = state.get("current_price", 0.0)

                        if current_price <= 0:
                            continue

                        should_review = False
                        review_reason = ""

                        # --- Deterministic SL/TP Check ---
                        if pos.side == "LONG":
                            if pos.current_stop > 0 and current_price <= pos.current_stop:
                                if pos.lifecycle_state == PositionState.UNMANAGED_ADOPTED:
                                    logger.info(
                                        "Emergency SL boundary reached on adopted position, forcing review",
                                        position_id=pos.position_id,
                                        symbol=pos.symbol,
                                        price=current_price,
                                        stop=pos.current_stop,
                                    )
                                    should_review = True
                                    review_reason = "emergency_sl_hit"
                                else:
                                    logger.info(
                                        "Stop loss hit (LONG)",
                                        position_id=pos.position_id,
                                        symbol=pos.symbol,
                                        price=current_price,
                                        stop=pos.current_stop,
                                    )
                                    await self._portfolio.update_position_state(
                                        pos.position_id, PositionState.CLOSING
                                    )
                                    await self._execution.execute_exit(pos, "SL_HIT")
                                    continue

                            if pos.current_target > 0 and current_price >= pos.current_target:
                                logger.info(
                                    "Take profit hit (LONG)",
                                    position_id=pos.position_id,
                                    symbol=pos.symbol,
                                    price=current_price,
                                    target=pos.current_target,
                                )
                                await self._portfolio.update_position_state(
                                    pos.position_id, PositionState.CLOSING
                                )
                                await self._execution.execute_exit(pos, "TP_HIT")
                                continue

                            # Track MFE/MAE
                            unrealized_pnl = (current_price - pos.avg_fill_price) * pos.quantity
                            if unrealized_pnl > pos.highest_unrealized_profit:
                                pos.highest_unrealized_profit = unrealized_pnl
                            drawdown = (pos.avg_fill_price - current_price) * pos.quantity
                            if drawdown > pos.maximum_drawdown:
                                pos.maximum_drawdown = drawdown
                        else:
                            if pos.current_stop > 0 and current_price >= pos.current_stop:
                                if pos.lifecycle_state == PositionState.UNMANAGED_ADOPTED:
                                    logger.info(
                                        "Emergency SL boundary reached on adopted position, forcing review",
                                        position_id=pos.position_id,
                                        symbol=pos.symbol,
                                        price=current_price,
                                        stop=pos.current_stop,
                                    )
                                    should_review = True
                                    review_reason = "emergency_sl_hit"
                                else:
                                    logger.info(
                                        "Stop loss hit (SHORT)",
                                        position_id=pos.position_id,
                                        symbol=pos.symbol,
                                        price=current_price,
                                        stop=pos.current_stop,
                                    )
                                    await self._portfolio.update_position_state(
                                        pos.position_id, PositionState.CLOSING
                                    )
                                    await self._execution.execute_exit(pos, "SL_HIT")
                                    continue

                            if pos.current_target > 0 and current_price <= pos.current_target:
                                logger.info(
                                    "Take profit hit (SHORT)",
                                    position_id=pos.position_id,
                                    symbol=pos.symbol,
                                    price=current_price,
                                    target=pos.current_target,
                                )
                                await self._portfolio.update_position_state(
                                    pos.position_id, PositionState.CLOSING
                                )
                                await self._execution.execute_exit(pos, "TP_HIT")
                                continue

                            unrealized_pnl = (pos.avg_fill_price - current_price) * pos.quantity
                            if unrealized_pnl > pos.highest_unrealized_profit:
                                pos.highest_unrealized_profit = unrealized_pnl
                            drawdown = (current_price - pos.avg_fill_price) * pos.quantity
                            if drawdown > pos.maximum_drawdown:
                                pos.maximum_drawdown = drawdown

                        # --- Intelligent Check ---
                        now = datetime.now(timezone.utc)
                        holding_hours = (now - pos.entry_timestamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600

                        if holding_hours >= REVIEW_HOLDING_HOURS:
                            should_review = True
                            review_reason = f"holding_time>{REVIEW_HOLDING_HOURS}h"

                        correlations = state.get("correlations", [])
                        avg_corr = 0.0
                        if correlations:
                            avg_corr = sum(abs(c.get("coefficient", 0)) for c in correlations) / len(correlations)
                        if avg_corr < REVIEW_CORRELATION_THRESHOLD:
                            should_review = True
                            review_reason += f" correlation_drop<{REVIEW_CORRELATION_THRESHOLD}"

                        if should_review:
                            is_adopted = pos.lifecycle_state == PositionState.UNMANAGED_ADOPTED

                            if not is_adopted:
                                pos.review_count += 1
                                await self._portfolio.update_position_state(
                                    pos.position_id,
                                    PositionState.UNDER_REVIEW,
                                    review_count=pos.review_count,
                                )
                            logger.info(
                                "Position sent to review",
                                position_id=pos.position_id,
                                reason=review_reason,
                                is_adopted=is_adopted,
                            )

                            try:
                                if is_adopted:
                                    review_prompt = (
                                        f"ADOPTED Position {pos.symbol} ({pos.side}): "
                                        f"current_price={current_price}, "
                                        f"emergency_stop={pos.current_stop}, "
                                        f"holding={holding_hours:.1f}h, "
                                        f"correlation={avg_corr:.3f}. "
                                        f"There is NO entry thesis. "
                                        f"Recommend HOLD or CLOSE."
                                    )
                                    system_prompt = UNMANAGED_ADOPTED_REVIEW_PROMPT
                                else:
                                    review_prompt = (
                                        f"Position {pos.symbol} ({pos.side}): "
                                        f"entry={pos.avg_fill_price}, "
                                        f"current={current_price}, "
                                        f"holding={holding_hours:.1f}h, "
                                        f"correlation={avg_corr:.3f}. "
                                        f"Recommend HOLD or CLOSE."
                                    )
                                    system_prompt = REVIEW_SYSTEM_PROMPT

                                response = await self._llm.request_completion(
                                    system_prompt=system_prompt,
                                    user_prompt=review_prompt,
                                )
                                review = json.loads(response.strip())
                                pos.current_recommendation = review.get("recommendation", "HOLD")

                                if review.get("recommendation") == "CLOSE":
                                    exit_reason = "EMERGENCY_SL_HIT" if is_adopted else "LLM_REVIEW_CLOSE"
                                    logger.info(
                                        "LLM recommended close",
                                        position_id=pos.position_id,
                                        reason=exit_reason,
                                    )
                                    await self._portfolio.update_position_state(
                                        pos.position_id,
                                        PositionState.CLOSING,
                                    )
                                    await self._execution.execute_exit(pos, exit_reason)
                            except Exception as e:
                                logger.error(
                                    "Review LLM call failed",
                                    position_id=pos.position_id,
                                    error=str(e),
                                )

                        # Persist updated MFE/MAE
                        self._portfolio._store.save_position(pos)

                    except Exception as e:
                        logger.error(
                            "Monitor check error for position",
                            position_id=pos.position_id,
                            symbol=pos.symbol,
                            error=str(e),
                        )

            except asyncio.CancelledError:
                logger.info("Position monitor loop cancelled")
                break
            except Exception as e:
                logger.error("Position monitor loop error", error=str(e))
                await asyncio.sleep(1)
