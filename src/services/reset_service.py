from __future__ import annotations

import structlog

from src.api.binance_client import BinanceClient
from src.core.events import EventBus
from src.core.models import SystemEvent
from src.db.portfolio_store import PortfolioStore
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("reset_service")


class EmergencyResetService:

    @staticmethod
    async def execute_hard_reset(
        binance_client: BinanceClient,
        portfolio_store: PortfolioStore,
        portfolio_manager: PortfolioManager,
        log,
    ) -> None:
        liquidated = 0
        liquidate_errors = 0

        open_positions = await binance_client.get_open_positions()
        if open_positions:
            log.warning(
                "Found open positions on exchange. Liquidating to sync state.",
                count=len(open_positions),
            )
            for pos in open_positions:
                try:
                    symbol = pos["symbol"]
                    amt = pos["position_amt"]
                    log.info("Liquidating position", symbol=symbol, amt=amt)
                    await binance_client.force_close_position(symbol, amt)
                    liquidated += 1
                except Exception as e:
                    log.error("Failed to liquidate position", symbol=pos["symbol"], error=str(e))
                    liquidate_errors += 1
        else:
            log.info("No open positions on exchange. Skipping liquidation.")

        closed_count = portfolio_store.reset_all_local_positions()
        log.warning(
            "Local positions reset to CLOSED",
            count=closed_count,
        )

        portfolio_manager.reload_from_database()

        event = SystemEvent(
            event_type="HARD_RESET_EXECUTED",
            service_name="EmergencyResetService",
            payload={
                "exchange_positions_liquidated": liquidated,
                "liquidate_errors": liquidate_errors,
                "local_positions_closed": closed_count,
            },
        )
        portfolio_store.append_audit_log(event)

        log.info(
            "Emergency reset complete",
            liquidated=liquidated,
            liquidate_errors=liquidate_errors,
            local_closed=closed_count,
        )
