from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal, ROUND_DOWN

import structlog

from src.api.binance_client import BinanceClient, BinanceClientError
from src.core.events import EventBus
from src.core.models import CandidateTrade, Position, SystemEvent

logger = structlog.get_logger("execution_service")

MAX_FILL_POLL_RETRIES = 5
FILL_POLL_INTERVAL = 1


class ExecutionService:
    def __init__(self, binance_client: BinanceClient, event_bus: EventBus, config: dict):
        self._client = binance_client
        self._event_bus = event_bus
        self._config = config
        self._event_bus.subscribe("CANDIDATE_APPROVED", self._on_candidate_approved)
        logger.info("ExecutionService initialized")

    async def _on_candidate_approved(self, event: SystemEvent) -> None:
        payload = event.payload
        candidate_data = payload.get("candidate", {})
        candidate = CandidateTrade(**candidate_data)
        asyncio.create_task(self._execute_entry(candidate))

    async def _round_quantity(self, symbol: str, raw_qty: float) -> str:
        step_size = await self._client.get_symbol_step_size(symbol)
        qty = Decimal(str(raw_qty))
        step = Decimal(str(step_size))
        precision = abs(step.as_tuple().exponent)
        valid_qty = (qty // step) * step
        return str(valid_qty.quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))

    async def _poll_authoritative_fill(self, symbol: str, client_order_id: str) -> dict:
        for attempt in range(MAX_FILL_POLL_RETRIES):
            if attempt > 0:
                await asyncio.sleep(FILL_POLL_INTERVAL)

            order = await self._client.get_order_status(
                symbol, orig_client_order_id=client_order_id
            )
            status = order.get("status", "")

            if status == "FILLED":
                fills = order.get("fills", [])
                total_commission = sum(
                    float(fill.get("commission", 0.0)) for fill in fills
                )
                self.logger.info(
                    "Authoritative fill data retrieved",
                    symbol=symbol,
                    order_id=order.get("orderId"),
                    status=status,
                    attempt=attempt + 1,
                )
                return {
                    "avgPrice": float(order.get("avgPrice", 0.0)),
                    "executedQty": float(order.get("executedQty", 0.0)),
                    "cumQuote": float(order.get("cumQuote", 0.0)),
                    "commission": total_commission,
                    "fills": fills,
                    "status": status,
                    "orderId": order.get("orderId"),
                }

            if status in ("CANCELED", "EXPIRED", "REJECTED"):
                logger.error(
                    "Order in terminal non-filled state",
                    symbol=symbol,
                    order_id=order.get("orderId"),
                    status=status,
                )
                raise BinanceClientError(
                    f"Order {client_order_id} for {symbol} entered state {status} "
                    f"after placement"
                )

            self.logger.info(
                "Order not yet filled, retrying",
                symbol=symbol,
                status=status,
                attempt=attempt + 1,
            )

        raise BinanceClientError(
            f"Order {client_order_id} for {symbol} not FILLED after "
            f"{MAX_FILL_POLL_RETRIES} retries"
        )

    @property
    def logger(self):
        return logger

    async def _execute_entry(self, candidate: CandidateTrade) -> None:
        try:
            exec_cfg = self._config.get("execution", {})
            sizing_mode = exec_cfg.get("sizing_mode", "risk_pct")
            sizing_value = float(exec_cfg.get("sizing_value", 2.0))
            leverage = int(exec_cfg.get("leverage", 10))

            stop_loss_pct = float(exec_cfg.get("stop_loss_pct", 0.98))
            take_profit_pct = float(exec_cfg.get("take_profit_pct", 1.04))

            raw_qty = (
                (sizing_value * leverage) / candidate.signal_strength
                if candidate.signal_strength > 0
                else 0
            )

            qty_str = await self._round_quantity(candidate.symbol, raw_qty)

            if float(qty_str) <= 0:
                logger.error(
                    "Quantity too small after rounding",
                    symbol=candidate.symbol,
                    raw_qty=raw_qty,
                )
                return

            side = candidate.proposed_side
            position_side = "BOTH"
            client_order_id = str(uuid.uuid4())

            await self._client.set_leverage(candidate.symbol, leverage)
            post_result = await self._client.place_market_order(
                symbol=candidate.symbol,
                side=side,
                quantity=qty_str,
                position_side=position_side,
                new_client_order_id=client_order_id,
            )

            order_id = post_result.get("orderId")
            returned_client_id = post_result.get("clientOrderId", client_order_id)
            poll_id = returned_client_id if returned_client_id else client_order_id

            auth = await self._poll_authoritative_fill(candidate.symbol, poll_id)

            avg_price = auth["avgPrice"]
            executed_qty = auth["executedQty"]
            commission = auth["commission"]

            if executed_qty <= 0 or avg_price <= 0:
                logger.error(
                    "Authoritative fill returned invalid values",
                    symbol=candidate.symbol,
                    avg_price=avg_price,
                    executed_qty=executed_qty,
                )
                return

            trade_side = "LONG" if side == "BUY" else "SHORT"
            if trade_side == "LONG":
                stop_loss = avg_price * stop_loss_pct
                take_profit = avg_price * take_profit_pct
            else:
                stop_loss = avg_price * (2.0 - stop_loss_pct)
                take_profit = avg_price * (2.0 - take_profit_pct)

            payload = {
                "type": "entry",
                "order_id": order_id,
                "client_order_id": client_order_id,
                "symbol": candidate.symbol,
                "side": trade_side,
                "avg_price": avg_price,
                "executed_qty": executed_qty,
                "commission": commission,
                "anchor_symbol": candidate.anchor_symbol,
                "correlation_score": candidate.correlation_score,
                "initial_stop_loss": stop_loss,
                "initial_take_profit": take_profit,
                "entry_thesis": (
                    f"Scanner signal: {candidate.signal_strength:.2f} confidence "
                    f"on {candidate.anchor_symbol}"
                ),
            }

            event = SystemEvent(
                event_type="ORDER_FILLED",
                service_name="ExecutionService",
                payload=payload,
            )
            await self._event_bus.publish(event)
            logger.info(
                "Entry order executed",
                symbol=candidate.symbol,
                side=trade_side,
                qty=executed_qty,
                price=avg_price,
                order_id=order_id,
                client_order_id=client_order_id,
            )

        except Exception as e:
            logger.error(
                "Entry execution failed",
                symbol=candidate.symbol,
                error=str(e),
            )

    async def execute_exit(self, position: Position, reason: str) -> None:
        try:
            side = "SELL" if position.side == "LONG" else "BUY"

            qty_str = await self._round_quantity(position.symbol, position.quantity)

            if float(qty_str) <= 0:
                logger.error(
                    "Exit quantity invalid after rounding",
                    symbol=position.symbol,
                    raw_qty=position.quantity,
                )
                return

            client_order_id = str(uuid.uuid4())

            post_result = await self._client.place_market_order(
                symbol=position.symbol,
                side=side,
                quantity=qty_str,
                position_side="BOTH",
                new_client_order_id=client_order_id,
            )

            returned_client_id = post_result.get("clientOrderId", client_order_id)
            poll_id = returned_client_id if returned_client_id else client_order_id

            auth = await self._poll_authoritative_fill(position.symbol, poll_id)

            avg_price = auth["avgPrice"]
            commission = auth["commission"]

            payload = {
                "type": "exit",
                "order_id": post_result.get("orderId"),
                "client_order_id": client_order_id,
                "position_id": position.position_id,
                "symbol": position.symbol,
                "side": position.side,
                "exit_price": avg_price,
                "commission": commission,
                "reason": reason,
            }

            event = SystemEvent(
                event_type="ORDER_FILLED",
                service_name="ExecutionService",
                payload=payload,
            )
            await self._event_bus.publish(event)
            logger.info(
                "Exit order executed",
                symbol=position.symbol,
                reason=reason,
                exit_price=avg_price,
                commission=commission,
                order_id=post_result.get("orderId"),
                client_order_id=client_order_id,
            )

        except Exception as e:
            logger.error(
                "Exit execution failed",
                symbol=position.symbol,
                error=str(e),
            )
