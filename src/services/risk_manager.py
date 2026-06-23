from __future__ import annotations

from datetime import datetime

import structlog

from src.core.models import CandidateTrade, SystemEvent
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("risk_manager")

MAX_CONCURRENT_POSITIONS = 3


class RiskManager:
    def __init__(self, max_positions: int = MAX_CONCURRENT_POSITIONS):
        self._max_positions = max_positions

    async def evaluate_candidate(self, candidate: CandidateTrade, portfolio: PortfolioManager) -> bool:
        open_positions = portfolio.get_open_positions()
        if len(open_positions) >= self._max_positions:
            logger.warning(
                "Risk rejected: max concurrent positions reached",
                current=len(open_positions),
                max=self._max_positions,
                symbol=candidate.symbol,
            )
            event = SystemEvent(
                event_type="RISK_REJECTED",
                service_name="RiskManager",
                payload={"reason": "max_positions", "symbol": candidate.symbol},
            )
            portfolio._store.append_audit_log(event)
            return False

        exposure = portfolio.get_exposure(candidate.symbol)
        if exposure > 0:
            logger.warning(
                "Risk rejected: duplicate symbol exposure",
                symbol=candidate.symbol,
                exposure=exposure,
            )
            event = SystemEvent(
                event_type="RISK_REJECTED",
                service_name="RiskManager",
                payload={"reason": "duplicate_symbol", "symbol": candidate.symbol},
            )
            portfolio._store.append_audit_log(event)
            return False

        return True
