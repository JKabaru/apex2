from __future__ import annotations

import structlog

from src.core.events import SystemEvent
from src.models.execution import (
    ExecutableTrade,
    ExecutedTrade,
    ExecutionStatus,
    TradeValidationReport,
    ValidationOutcome,
    ValidationOutcomeStatus,
)

logger = structlog.get_logger("protection_coordinator")


class ProtectionCoordinatorError(Exception):
    pass


class CriticalProtectionFailure(ProtectionCoordinatorError):
    pass


class ProtectionCoordinator:
    def __init__(self, live_executor, event_bus, config: dict):
        self._live = live_executor
        self._event_bus = event_bus
        self._config = config

    async def place_and_verify(
        self,
        trade: ExecutableTrade,
        executed_qty: float,
        avg_price: float,
    ) -> dict:
        side = "SELL" if trade.trade_side == "LONG" else "BUY"

        protection_data = await self._live.place_protection(
            symbol=trade.symbol,
            side=side,
            stop_price=trade.stop_price,
            tp_price=trade.tp_price,
            position_id=trade.execution_id,
            quantity=executed_qty,
            current_price=avg_price,
        )

        if not self._verify_ownership(protection_data, trade):
            raise CriticalProtectionFailure(
                f"Protection ownership verification failed for {trade.symbol} "
                f"execution {trade.execution_id}: client order IDs do not match"
            )

        logger.info(
            "PROTECTION_VERIFIED_OWNERSHIP",
            symbol=trade.symbol,
            execution_id=trade.execution_id,
            stop_client_id=protection_data.get("stop_client_order_id"),
            tp_client_id=protection_data.get("tp_client_order_id"),
        )

        return protection_data

    @staticmethod
    def _short_id(position_id: str) -> str:
        return position_id.replace("-", "")[:16]

    def _verify_ownership(self, protection_data: dict, trade: ExecutableTrade) -> bool:
        sid = self._short_id(trade.execution_id)
        expected_stop_cid = f"SL_{sid}"
        expected_tp_cid = f"TP_{sid}"

        actual_stop_cid = protection_data.get("stop_client_order_id", "")
        actual_tp_cid = protection_data.get("tp_client_order_id", "")

        if actual_stop_cid != expected_stop_cid:
            logger.error(
                "PROTECTION_OWNERSHIP_MISMATCH",
                field="stop_client_order_id",
                expected=expected_stop_cid,
                actual=actual_stop_cid,
            )
            return False

        if actual_tp_cid != expected_tp_cid:
            logger.error(
                "PROTECTION_OWNERSHIP_MISMATCH",
                field="tp_client_order_id",
                expected=expected_tp_cid,
                actual=actual_tp_cid,
            )
            return False

        return True
