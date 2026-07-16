from __future__ import annotations

import math
from datetime import datetime
from statistics import mean, stdev
from typing import Optional

import structlog

from src.core.events import EventBus
from src.core.models import Position, PositionState, SystemEvent
from src.db.portfolio_store import PortfolioStore

logger = structlog.get_logger("metrics_service")


def compute_realized_pnl(pos: Position) -> Optional[float]:
    if pos.exit_price is None:
        return None
    direction = 1.0 if pos.side.upper() == "LONG" else -1.0
    gross = direction * (pos.exit_price - pos.avg_fill_price) * pos.quantity
    total_fees = (pos.fees or 0.0) + (pos.exit_fees or 0.0)
    return round(gross - total_fees, 2)


class MetricsService:
    def __init__(self, event_bus: EventBus, store: PortfolioStore):
        self._store = store
        self._event_bus = event_bus
        self._event_bus.subscribe("POSITION_UPDATED", self._on_position_updated)

        self._fast_win_count = 0
        self._fast_loss_count = 0
        self._fast_total_pnl = 0.0
        self._fast_total_fees = 0.0
        self._pnl_samples: list[float] = []
        logger.info("MetricsService initialized")

    async def _on_position_updated(self, event: SystemEvent) -> None:
        payload = event.payload
        new_state = payload.get("new_state", "")
        if new_state not in ("CLOSED", "ARCHIVED"):
            return

        position_id = payload.get("position_id", "")
        pos = self._store.get_position_by_id(position_id)
        if pos is None:
            return

        pnl = compute_realized_pnl(pos)
        if pnl is None:
            return

        self._fast_total_pnl += pnl
        self._fast_total_fees += (pos.fees or 0.0) + (pos.exit_fees or 0.0)
        self._pnl_samples.append(pnl)
        if pnl >= 0:
            self._fast_win_count += 1
        else:
            self._fast_loss_count += 1

        logger.debug(
            "Fast metrics updated",
            position_id=position_id, pnl=pnl,
            total_pnl=round(self._fast_total_pnl, 2),
        )

    def get_fast_metrics(self) -> dict:
        total = self._fast_win_count + self._fast_loss_count
        return {
            "win_count": self._fast_win_count,
            "loss_count": self._fast_loss_count,
            "total_trades": total,
            "win_rate": round(self._fast_win_count / total * 100, 1) if total > 0 else 0.0,
            "total_realized_pnl": round(self._fast_total_pnl, 2),
            "total_fees": round(self._fast_total_fees, 2),
        }

    def compute_slow_metrics(self) -> dict:
        positions = self._store.get_completed_positions()
        open_count = self._store.get_open_position_count()
        mode_counts = self._store.get_open_positions_by_mode()

        wins = []
        losses = []
        total_pnl = 0.0
        total_fees = 0.0
        equity_curve = [0.0]
        running_pnl = 0.0

        for pos in positions:
            pnl = compute_realized_pnl(pos)
            if pnl is None:
                continue
            total_pnl += pnl
            fees = (pos.fees or 0.0) + (pos.exit_fees or 0.0)
            total_fees += fees
            if pnl >= 0:
                wins.append(pnl)
            else:
                losses.append(pnl)
            running_pnl += pnl
            equity_curve.append(running_pnl)

        win_count = len(wins)
        loss_count = len(losses)
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        avg_win = mean(wins) if wins else 0.0
        avg_loss = mean(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        max_dd = self._compute_max_drawdown(equity_curve)
        sharpe = self._compute_sharpe_from_pnls(self._pnl_samples + list(total_pnl for _ in range(max(0, 2 - len(self._pnl_samples)))))

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "portfolio_value": round(total_pnl + 10000.0, 2),
            "total_realized_pnl": round(total_pnl, 2),
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 0.0,
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd * 100, 2),
            "open_positions": open_count,
            "live_positions": mode_counts.get("LIVE", 0),
            "shadow_positions": mode_counts.get("SHADOW", 0),
            "total_fees": round(total_fees, 2),
        }

    def record_slow_metrics(self) -> int:
        metrics = self.compute_slow_metrics()
        snapshot_id = self._store.save_metrics_snapshot(metrics)
        logger.info("Slow metrics snapshot recorded", snapshot_id=snapshot_id, **metrics)
        return snapshot_id

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _compute_sharpe_from_pnls(pnls: list[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        mean_ret = mean(pnls)
        std_ret = stdev(pnls)
        if std_ret == 0.0:
            return 0.0
        return math.sqrt(len(pnls)) * mean_ret / std_ret
