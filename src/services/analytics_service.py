from __future__ import annotations

import structlog

from src.core.events import EventBus
from src.core.models import Position, PositionState, SystemEvent
from src.db.portfolio_store import PortfolioStore

logger = structlog.get_logger("analytics_service")
rs_logger = structlog.get_logger("research")


class AnalyticsService:
    def __init__(self, event_bus: EventBus, store: PortfolioStore):
        self._store = store
        self._event_bus = event_bus
        self._event_bus.subscribe("CALIBRATION_COMPUTED", self._on_calibration_computed)
        self._event_bus.subscribe("TRADE_COMPLETED", self._on_trade_completed)
        self._event_bus.subscribe("CANDIDATE_EVALUATED", self._on_candidate_evaluated)

        self._live_trade_count = 0
        self._shadow_trade_count = 0
        self._rejected_opportunities = 0
        self._constraint_breakdown: dict[str, int] = {}
        self._missed_alpha: float = 0.0
        self._shadow_wins = 0
        self._shadow_losses = 0
        self._calibration_samples: list[dict] = []
        logger.info("AnalyticsService initialized")

    async def _on_calibration_computed(self, event: SystemEvent) -> None:
        payload = event.payload
        rs_logger.info(
            "Calibration computed",
            live_position_id=payload.get("live_position_id"),
            mirror_position_id=payload.get("mirror_position_id"),
            calibration_data=payload.get("calibration_data"),
            calibration_model=payload.get("calibration_model", "v1"),
            calibration_version=payload.get("calibration_version", "1.0"),
        )
        if payload.get("calibration_data"):
            self._calibration_samples.append(payload["calibration_data"])

    async def _on_trade_completed(self, event: SystemEvent) -> None:
        payload = event.payload
        exec_mode = payload.get("execution_mode", "LIVE")
        origin = payload.get("origin", "NORMAL")
        pnl = payload.get("pnl", 0.0)

        if origin == "MIRROR":
            return

        if exec_mode == "LIVE":
            self._live_trade_count += 1
        elif exec_mode == "SHADOW":
            self._shadow_trade_count += 1
            if pnl > 0:
                self._shadow_wins += 1
            else:
                self._shadow_losses += 1
            if payload.get("origin") == "CONSTRAINT" and pnl > 0:
                self._missed_alpha += pnl

        rs_logger.info(
            "Trade completed",
            position_id=payload.get("position_id"),
            execution_mode=exec_mode,
            origin=origin,
            pnl=pnl,
            exit_reason=payload.get("exit_reason"),
        )

    async def _on_candidate_evaluated(self, event: SystemEvent) -> None:
        payload = event.payload
        risk_decision = payload.get("risk_decision", "")
        risk_reason = payload.get("risk_decision_reason", "")

        if risk_decision in ("REJECTED_CONSTRAINT", "REJECTED_QUALITY", "DEFERRED"):
            self._rejected_opportunities += 1
            if risk_reason:
                self._constraint_breakdown[risk_reason] = (
                    self._constraint_breakdown.get(risk_reason, 0) + 1
                )

    def get_metrics(self) -> dict:
        avg_calibration = {}
        if self._calibration_samples:
            keys = self._calibration_samples[0].keys()
            for key in keys:
                values = [s.get(key, 0) for s in self._calibration_samples if isinstance(s.get(key), (int, float))]
                if values:
                    avg_calibration[key] = sum(values) / len(values)

        shadow_total = self._shadow_wins + self._shadow_losses
        shadow_win_rate = (self._shadow_wins / shadow_total * 100) if shadow_total > 0 else 0.0

        return {
            "live_trades": self._live_trade_count,
            "shadow_trades": self._shadow_trade_count,
            "rejected_opportunities": self._rejected_opportunities,
            "constraint_breakdown": dict(self._constraint_breakdown),
            "missed_alpha_usdt": round(self._missed_alpha, 2),
            "shadow_win_rate_pct": round(shadow_win_rate, 1),
            "shadow_wins": self._shadow_wins,
            "shadow_losses": self._shadow_losses,
            "calibration_samples": len(self._calibration_samples),
            "avg_calibration_errors": avg_calibration,
        }
