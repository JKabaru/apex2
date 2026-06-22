import asyncio
import traceback

import structlog

from ..api.binance_client import BinanceClient
from .router import round_step_size
from .trade_manager import TradeManager

logger = structlog.get_logger("trade_monitor")


async def monitor_open_trades(
    binance_client: BinanceClient,
    trade_manager: TradeManager,
    symbol_prices: dict,
):
    logger.info("Trade monitor started")
    while True:
        try:
            trades = await trade_manager.get_open_trades()
            for trade in trades:
                try:
                    current_price = symbol_prices.get(trade["symbol"])
                    if current_price is None:
                        continue

                    if trade["stop_loss"] <= 0 or trade["take_profit"] <= 0:
                        logger.warning(
                            "Trade has invalid SL/TP (0.0), skipping close check",
                            trade_id=trade["trade_id"],
                            symbol=trade["symbol"],
                        )
                        continue

                    should_close = False
                    close_side = ""
                    reason = ""

                    if trade["side"] == "LONG":
                        if current_price <= trade["stop_loss"]:
                            should_close = True
                            close_side = "SELL"
                            reason = "SL"
                        elif current_price >= trade["take_profit"]:
                            should_close = True
                            close_side = "SELL"
                            reason = "TP"
                    else:
                        if current_price >= trade["stop_loss"]:
                            should_close = True
                            close_side = "BUY"
                            reason = "SL"
                        elif current_price <= trade["take_profit"]:
                            should_close = True
                            close_side = "BUY"
                            reason = "TP"

                    if not should_close:
                        continue

                    logger.info(
                        "Closing trade",
                        trade_id=trade["trade_id"],
                        symbol=trade["symbol"],
                        reason=reason,
                        current_price=current_price,
                        stop_loss=trade["stop_loss"],
                        take_profit=trade["take_profit"],
                    )

                    try:
                        step_size = await binance_client.get_symbol_step_size(trade["symbol"])
                    except Exception as e:
                        logger.error("Failed to retrieve dynamic step size for close, using fallback", symbol=trade["symbol"], error=str(e))
                        step_size = 0.001 if any(x in trade["symbol"].upper() for x in ["BTC", "ETH"]) else 0.01

                    close_qty = round_step_size(float(trade["position_size"]), step_size)
                    close_result = await binance_client.place_market_order(
                        symbol=trade["symbol"],
                        side=close_side,
                        quantity=close_qty,
                        position_side="BOTH",
                    )

                    exit_price = float(close_result.get("avgPrice", current_price))
                    commission = float(close_result.get("commission", 0.0))

                    if trade["side"] == "LONG":
                        realized_pnl = (exit_price - trade["entry_price"]) * trade["position_size"]
                    else:
                        realized_pnl = (trade["entry_price"] - exit_price) * trade["position_size"]

                    await trade_manager.close_trade(
                        trade_id=trade["trade_id"],
                        exit_price=exit_price,
                        fees=commission,
                        pnl=realized_pnl,
                        reason=reason,
                    )

                    logger.info(
                        "Trade closed successfully",
                        trade_id=trade["trade_id"],
                        exit_price=exit_price,
                        pnl=realized_pnl,
                        fees=commission,
                        reason=reason,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process trade in monitor",
                        trade_id=trade.get("trade_id", "unknown"),
                        error=str(e),
                        traceback=traceback.format_exc(),
                    )

        except Exception as e:
            logger.error(
                "Trade monitor cycle error",
                error=str(e),
                traceback=traceback.format_exc(),
            )

        await asyncio.sleep(10)
