import math
import traceback
from decimal import Decimal, ROUND_DOWN

import structlog

from ..agent.llm_executor import AgentDecision
from ..api.binance_client import BinanceClient
from .trade_manager import TradeManager

logger = structlog.get_logger("execution_router")


def round_step_size(quantity: float, step_size: float = 0.001) -> str:
    qty = Decimal(str(quantity))
    step = Decimal(str(step_size))
    precision = abs(step.as_tuple().exponent)
    valid_qty = (qty // step) * step
    return str(valid_qty.quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))


async def execute_trade_decision(
    decision: AgentDecision,
    symbol: str,
    current_price: float,
    account_balance: float,
    config: dict,
    binance_client: BinanceClient,
    trade_manager: TradeManager,
) -> dict:
    try:
        if decision.action == "HOLD":
            logger.info("Trade decision HOLD — ignored")
            return {"status": "IGNORED"}

        exec_cfg = config.get("execution", {})
        exec_mode = exec_cfg.get("mode")
        if exec_mode not in ("testnet", "live"):
            raise ValueError(f"FATAL: Invalid execution mode '{exec_mode}'. Must be 'testnet' or 'live'.")
        logger.info("Execution mode validated", mode=exec_mode)

        sizing_mode = exec_cfg.get("sizing_mode", "risk_pct")
        sizing_value = float(exec_cfg.get("sizing_value", 2.0))
        leverage = int(exec_cfg.get("leverage", 10))

        if sizing_mode == "risk_pct":
            risk_amount = account_balance * (sizing_value / 100.0)
        else:
            risk_amount = sizing_value

        try:
            step_size = await binance_client.get_symbol_step_size(symbol)
        except Exception as e:
            logger.error("Failed to retrieve dynamic step size, aborting trade", symbol=symbol, error=str(e))
            return {"status": "FAILED", "error": f"Failed to retrieve dynamic step size: {e}"}

        raw_qty = (risk_amount * leverage) / current_price
        qty = round_step_size(raw_qty, step_size)

        if float(qty) <= 0:
            logger.error("Quantity too small, aborting trade", raw_qty=raw_qty, qty=qty, step_size=step_size)
            return {"status": "FAILED", "error": "Quantity too small after rounding"}

        if decision.action == "BUY":
            side = "BUY"
            position_side = "BOTH"
            trade_side = "LONG"
        else:
            side = "SELL"
            position_side = "BOTH"
            trade_side = "SHORT"

        await binance_client.set_leverage(symbol, leverage)

        logger.info(
            "Placing market order",
            symbol=symbol,
            side=side,
            position_side=position_side,
            qty=qty,
            leverage=leverage,
            sizing_mode=sizing_mode,
            sizing_value=sizing_value,
        )

        order_result = await binance_client.place_market_order(
            symbol=symbol,
            side=side,
            quantity=qty,
            position_side=position_side,
        )

        fill_price = float(order_result.get("avgPrice", current_price))
        executed_qty = float(order_result.get("executedQty", qty))
        commission = float(order_result.get("commission", 0.0))

        if executed_qty <= 0:
            logger.error("Order filled with zero quantity", symbol=symbol)
            return {"status": "FAILED", "error": "Order filled with zero quantity"}

        if trade_side == "LONG":
            stop_loss = fill_price * 0.98
            take_profit = fill_price * 1.04
        else:
            stop_loss = fill_price * 1.02
            take_profit = fill_price * 0.96

        trade_data = {
            "symbol": symbol,
            "side": trade_side,
            "entry_price": fill_price,
            "position_size": executed_qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        trade_id = await trade_manager.open_trade(trade_data)

        logger.info(
            "Trade executed",
            trade_id=trade_id,
            symbol=symbol,
            side=trade_side,
            fill_price=fill_price,
            qty=executed_qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees=commission,
        )

        return {
            "status": "EXECUTED",
            "trade_id": trade_id,
            "fill_price": fill_price,
            "position_size": executed_qty,
            "sl": stop_loss,
            "tp": take_profit,
        }

    except Exception as e:
        logger.error(
            "Trade execution failed",
            symbol=symbol,
            action=decision.action,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        return {"status": "FAILED", "error": str(e)}
