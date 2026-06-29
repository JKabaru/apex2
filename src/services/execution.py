from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_DOWN
from datetime import datetime

import structlog

from src.api.binance_client import BinanceClient, BinanceClientError
from src.core.events import EventBus
from src.core.models import ExecutionContext, Position, ProtectionOrders, SystemEvent, VirtualFill
from src.services.market_context import MarketContextService
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("execution_service")

MAX_FILL_POLL_RETRIES = 5
PROTECTION_RETRIES = 3


class InsufficientMarginError(Exception):
    pass


class CriticalProtectionFailure(Exception):
    pass

FILL_POLL_INTERVAL = 1


class IExecutor(ABC):
    @abstractmethod
    async def execute_entry(
        self, symbol: str, side: str, quantity_str: str, context: ExecutionContext
    ) -> dict:
        ...

    @abstractmethod
    async def execute_exit(
        self, symbol: str, side: str, quantity_str: str, position: Position
    ) -> dict:
        ...


class LiveExecutor(IExecutor):
    def __init__(self, client: BinanceClient, config: dict):
        self._client = client
        self._config = config
        logger.info("LiveExecutor initialized")

    async def round_quantity(self, symbol: str, raw_qty: float) -> str:
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
                logger.info(
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

            logger.info(
                "Order not yet filled, retrying",
                symbol=symbol,
                status=status,
                attempt=attempt + 1,
            )

        raise BinanceClientError(
            f"Order {client_order_id} for {symbol} not FILLED after "
            f"{MAX_FILL_POLL_RETRIES} retries"
        )

    async def execute_entry(
        self, symbol: str, side: str, quantity_str: str, context: ExecutionContext
    ) -> dict:
        position_side = "BOTH"
        client_order_id = str(uuid.uuid4())

        await self._client.set_leverage(symbol, int(self._config.get("execution", {}).get("leverage", 10)))
        post_result = await self._client.place_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity_str,
            position_side=position_side,
            new_client_order_id=client_order_id,
        )

        order_id = post_result.get("orderId")
        returned_client_id = post_result.get("clientOrderId", client_order_id)
        poll_id = returned_client_id if returned_client_id else client_order_id

        return await self._poll_authoritative_fill(symbol, poll_id)

    async def execute_exit(
        self, symbol: str, side: str, quantity_str: str, position: Position
    ) -> dict:
        try:
            await self.cancel_all_orders(symbol)
            logger.info("Cancelled all open orders before exit to prevent orphaned STOP_MARKET", symbol=symbol)
        except Exception as e:
            logger.warning("Failed to cancel open orders before exit", symbol=symbol, error=str(e))

        client_order_id = str(uuid.uuid4())

        post_result = await self._client.place_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity_str,
            position_side="BOTH",
            new_client_order_id=client_order_id,
        )

        returned_client_id = post_result.get("clientOrderId", client_order_id)
        poll_id = returned_client_id if returned_client_id else client_order_id

        return await self._poll_authoritative_fill(symbol, poll_id)

    @staticmethod
    def _short_id(position_id: str) -> str:
        return position_id.replace("-", "")[:16]

    async def place_protection(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        tp_price: float,
        position_id: str,
        quantity: float,
        current_price: float,
    ) -> dict:
        sid = self._short_id(position_id)
        stop_client_id = f"SL_{sid}"
        tp_client_id = f"TP_{sid}"
        last_error = None

        for attempt in range(PROTECTION_RETRIES):
            try:
                stop_data = await self._client.place_algo_stop(
                    symbol, side, stop_price, position_id,
                    client_algo_id=stop_client_id,
                    estimated_qty=quantity, current_price=current_price,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "Stop placement attempt failed, retrying",
                    symbol=symbol, attempt=attempt + 1, error=str(e),
                )
                await asyncio.sleep(1 * (attempt + 1))

        if last_error is not None:
            raise CriticalProtectionFailure(
                f"Stop placement failed after {PROTECTION_RETRIES} retries "
                f"for {symbol} position {position_id}: {last_error}"
            )

        for attempt in range(PROTECTION_RETRIES):
            try:
                tp_data = await self._client.place_algo_tp(
                    symbol, side, tp_price, position_id,
                    client_algo_id=tp_client_id,
                    estimated_qty=quantity, current_price=current_price,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "TP placement attempt failed, retrying",
                    symbol=symbol, attempt=attempt + 1, error=str(e),
                )
                await asyncio.sleep(1 * (attempt + 1))

        if last_error is not None:
            try:
                await self._client.cancel_algo_by_client_id(symbol, stop_client_id)
                logger.info("Cancelled stop after TP failure", symbol=symbol, client_id=stop_client_id)
            except Exception:
                pass
            raise CriticalProtectionFailure(
                f"TP placement failed after {PROTECTION_RETRIES} retries "
                f"for {symbol} position {position_id}: {last_error}"
            )

        return {
            "stop_order_id": stop_data.get("algoId"),
            "stop_client_order_id": stop_client_id,
            "tp_order_id": tp_data.get("algoId"),
            "tp_client_order_id": tp_client_id,
            "stop_price": stop_price,
            "tp_price": tp_price,
        }

    async def update_trailing_stop(
        self,
        symbol: str,
        side: str,
        new_stop_price: float,
        position_id: str,
        old_stop_client_order_id: str = None,
        quantity: float = None,
        current_price: float = None,
    ) -> dict:
        sid = self._short_id(position_id)
        ts = int(time.time() * 1000)
        new_client_id = f"ST_{sid}_{ts}"
        new_data = await self._client.place_algo_stop(
            symbol, side, new_stop_price, position_id,
            client_algo_id=new_client_id,
            estimated_qty=quantity, current_price=current_price,
        )

        if old_stop_client_order_id:
            try:
                await self._client.cancel_algo_by_client_id(symbol, old_stop_client_order_id)
                logger.info(
                    "Old trailing stop cancelled",
                    symbol=symbol, client_id=old_stop_client_order_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to cancel old trailing stop — it will auto-expire",
                    symbol=symbol, client_id=old_stop_client_order_id, error=str(e),
                )

        return {
            "order_id": new_data.get("algoId"),
            "client_order_id": new_client_id,
            "stop_price": new_stop_price,
        }

    async def cancel_protection(self, symbol: str, stop_client_order_id: str = None, tp_client_order_id: str = None) -> None:
        if stop_client_order_id:
            try:
                await self._client.cancel_algo_by_client_id(symbol, stop_client_order_id)
                logger.info("Stop protection cancelled", symbol=symbol, client_id=stop_client_order_id)
            except Exception as e:
                logger.warning("Failed to cancel stop protection", symbol=symbol, error=str(e))
        if tp_client_order_id:
            try:
                await self._client.cancel_algo_by_client_id(symbol, tp_client_order_id)
                logger.info("TP protection cancelled", symbol=symbol, client_id=tp_client_order_id)
            except Exception as e:
                logger.warning("Failed to cancel TP protection", symbol=symbol, error=str(e))

    async def cancel_all_orders(self, symbol: str) -> None:
        await self._client.cancel_all_open_orders(symbol)

    async def get_available_balance(self, asset: str = "USDT") -> float:
        info = await self._client.get_account_info()
        for a in info.get("assets", []):
            if a["asset"] == asset:
                return float(a.get("availableBalance", 0.0))
        return 0.0

    async def get_symbol_taker_fee(self, symbol: str) -> float:
        try:
            rate_data = await self._client.get_commission_rate(symbol)
            return float(rate_data.get("takerCommissionRate", 0.0005))
        except Exception:
            logger.warning(
                "Failed to fetch taker fee, using default 0.05%",
                symbol=symbol,
            )
            return 0.0005


class VirtualExecutor(IExecutor):
    def __init__(self, market_context: MarketContextService, config: dict):
        self._context = market_context
        self._config = config
        logger.info("VirtualExecutor initialized")

    async def _get_synthetic_price(
        self, symbol: str, is_buy: bool, context: ExecutionContext
    ) -> tuple[float, float, float, float]:
        state = await self._context.get_state(symbol)
        market_price = state.get("current_price", 0.0)
        if market_price <= 0:
            logger.error("No market price available for shadow execution", symbol=symbol)
            raise ValueError(f"No market price for {symbol}")

        params = context.execution_parameters
        slippage_bps = float(params.get("slippage_bps", 3.0))
        spread_bps = float(params.get("spread_bps", 2.0))
        fee_bps = float(params.get("fee_bps", 4.0))

        direction = 1.0 if is_buy else -1.0
        synthetic_price = market_price * (
            1.0 + direction * (slippage_bps + spread_bps / 2.0) / 10000.0
        )

        drift_bps = round((synthetic_price - market_price) / market_price * 10000, 2) if market_price else 0.0
        logger.info(
            "Shadow synthetic fill",
            symbol=symbol, side="BUY" if is_buy else "SELL",
            market_price=market_price, synthetic_price=round(synthetic_price, 4),
            drift_bps=drift_bps, slippage_bps=slippage_bps, spread_bps=spread_bps, fee_bps=fee_bps,
        )
        return synthetic_price, slippage_bps, spread_bps, fee_bps

    async def execute_entry(
        self, symbol: str, side: str, quantity_str: str, context: ExecutionContext
    ) -> dict:
        is_buy = side == "BUY"
        synthetic_price, slippage_bps, spread_bps, fee_bps = (
            await self._get_synthetic_price(symbol, is_buy, context)
        )
        qty = float(quantity_str)
        fee_cost = qty * synthetic_price * fee_bps / 10000.0

        return {
            "avgPrice": synthetic_price,
            "executedQty": qty,
            "cumQuote": qty * synthetic_price,
            "commission": fee_cost,
            "fills": [],
            "status": "FILLED",
            "orderId": f"shadow_{uuid.uuid4()}",
            "synthetic_slippage_bps": slippage_bps,
            "synthetic_spread_bps": spread_bps,
            "synthetic_fee_bps": fee_bps,
        }

    async def execute_exit(
        self, symbol: str, side: str, quantity_str: str, position: Position
    ) -> dict:
        is_buy = side == "BUY"

        params = position.execution_parameters or {}
        slippage_bps = float(params.get("slippage_bps", 3.0))
        spread_bps = float(params.get("spread_bps", 2.0))
        fee_bps = float(params.get("fee_bps", 4.0))

        state = await self._context.get_state(symbol)
        market_price = state.get("current_price", 0.0)
        if market_price <= 0:
            logger.error("No market price for shadow exit", symbol=symbol)
            raise ValueError(f"No market price for {symbol}")

        direction = -1.0 if position.side == "LONG" else 1.0
        synthetic_price = market_price * (
            1.0 + direction * (slippage_bps + spread_bps / 2.0) / 10000.0
        )
        qty = float(quantity_str)
        fee_cost = qty * synthetic_price * fee_bps / 10000.0

        return {
            "avgPrice": synthetic_price,
            "executedQty": qty,
            "cumQuote": qty * synthetic_price,
            "commission": fee_cost,
            "status": "FILLED",
            "orderId": f"shadow_exit_{uuid.uuid4()}",
        }


class ExecutionService:
    def __init__(
        self,
        live_executor: LiveExecutor,
        virtual_executor: VirtualExecutor,
        market_context: MarketContextService,
        portfolio_mgr: PortfolioManager,
        event_bus: EventBus,
        config: dict,
        mirror_enabled: bool = True,
    ):
        self._live = live_executor
        self._virtual = virtual_executor
        self._context = market_context
        self._portfolio = portfolio_mgr
        self._event_bus = event_bus
        self._config = config
        self._mirror_enabled = mirror_enabled
        self._event_bus.subscribe("EXECUTE_TRADE", self._on_execute_trade)
        logger.info("ExecutionService initialized", mirror_enabled=mirror_enabled)

    async def _on_execute_trade(self, event: SystemEvent) -> None:
        payload = event.payload
        context_data = payload.get("context", {})
        context = ExecutionContext(**context_data)
        asyncio.create_task(self.execute_entry(context))

    async def _compute_sizing(
        self, context: ExecutionContext
    ) -> tuple[str, float]:
        exec_cfg = self._config.get("execution", {})
        sizing_mode = exec_cfg.get("sizing_mode", "risk_pct")
        sizing_value = float(exec_cfg.get("sizing_value", 2.0))
        leverage = int(exec_cfg.get("leverage", 10))

        confidence = context.llm_confidence if context.llm_confidence > 0 else 0.5
        raw_qty = (sizing_value * leverage) / confidence

        if context.execution_mode == "LIVE":
            qty_str = await self._live.round_quantity(context.symbol, raw_qty)
        else:
            qty_str = f"{round(raw_qty, 4):.4f}"
        qty = float(qty_str)
        if qty <= 0:
            logger.error(
                "Quantity too small after rounding",
                symbol=context.symbol,
                raw_qty=raw_qty,
            )
            return "0", 0.0
        return qty_str, qty

    async def execute_entry(self, context: ExecutionContext) -> None:
        try:
            qty_str, qty = await self._compute_sizing(context)
            if qty <= 0:
                return

            side = context.side
            trade_side = "LONG" if side == "BUY" else "SHORT"

            if context.execution_mode == "LIVE":
                # ── Pre-trade exchange validations ──
                exec_cfg = self._config.get("execution", {})
                state = await self._context.get_state(context.symbol)
                current_price = state.get("current_price", 0.0)

                # Minimum notional (5 USDT)
                min_notional = 5.0
                notional = qty * current_price
                if notional < min_notional and current_price > 0:
                    adjusted_qty = min_notional / current_price
                    qty_str = await self._live.round_quantity(context.symbol, adjusted_qty)
                    qty = float(qty_str)
                    if qty <= 0:
                        logger.error("Adjusted quantity too small after rounding", symbol=context.symbol)
                        return
                    notional = qty * current_price
                    logger.warning(
                        "Position size adjusted to meet minimum notional",
                        symbol=context.symbol, new_qty=qty, notional=round(notional, 2),
                    )

                # Available balance check including fees and slippage buffer
                leverage = int(exec_cfg.get("leverage", 10))
                taker_fee_rate = await self._live.get_symbol_taker_fee(context.symbol)
                notional_value = qty * current_price
                initial_margin = notional_value / leverage
                open_fee = notional_value * taker_fee_rate
                slippage_buffer = 1.01
                required_margin = (initial_margin + open_fee) * slippage_buffer
                margin_asset = "USDC" if context.symbol.endswith("USDC") else "USDT"
                available_balance = await self._live.get_available_balance(asset=margin_asset)
                if required_margin > available_balance:
                    raise InsufficientMarginError(
                        f"Cannot execute entry. Required: {required_margin:.4f} {margin_asset}, "
                        f"Available: {available_balance:.4f} {margin_asset} (Includes fee buffer)"
                    )
                auth = await self._live.execute_entry(
                    context.symbol, side, qty_str, context
                )
            else:
                auth = await self._virtual.execute_entry(
                    context.symbol, side, qty_str, context
                )

            avg_price = auth["avgPrice"]
            executed_qty = auth["executedQty"]
            commission = auth.get("commission", 0.0)

            if executed_qty <= 0 or avg_price <= 0:
                logger.error(
                    "Fill returned invalid values",
                    symbol=context.symbol,
                    avg_price=avg_price,
                    executed_qty=executed_qty,
                )
                return

            exec_cfg = self._config.get("execution", {})
            stop_loss_pct = float(exec_cfg.get("stop_loss_pct", 0.98))
            take_profit_pct = float(exec_cfg.get("take_profit_pct", 1.04))

            if trade_side == "LONG":
                stop_loss = avg_price * stop_loss_pct
                take_profit = avg_price * take_profit_pct
            else:
                stop_loss = avg_price * (2.0 - stop_loss_pct)
                take_profit = avg_price * (2.0 - take_profit_pct)

            protection_data = None
            if context.execution_mode == "LIVE":
                hard_stop_side = "SELL" if trade_side == "LONG" else "BUY"
                try:
                    protection_data = await self._live.place_protection(
                        context.symbol, hard_stop_side,
                        stop_loss, take_profit,
                        context.execution_id,
                        executed_qty, avg_price,
                    )
                    logger.info(
                        "Exchange protection placed",
                        symbol=context.symbol,
                        stop_price=stop_loss, tp_price=take_profit,
                    )
                except CriticalProtectionFailure as e:
                    logger.critical(
                        "CRITICAL_PROTECTION_FAILURE: Hard stop placement failed. "
                        "Immediately flattening position to prevent naked capital exposure.",
                        symbol=context.symbol, error=str(e),
                    )
                    event = SystemEvent(
                        event_type="PROTECTION_FAILED",
                        service_name="ExecutionService",
                        payload={
                            "symbol": context.symbol,
                            "execution_id": context.execution_id,
                            "error": str(e),
                            "emergency_close": True,
                        },
                    )
                    await self._event_bus.publish(event)
                    # EMERGENCY CLOSE: prevent naked position on exchange
                    try:
                        await self._live.cancel_all_orders(context.symbol)
                        await self._live._client.force_close_position(
                            context.symbol, executed_qty if side == "BUY" else -executed_qty,
                        )
                        logger.info(
                            "Emergency position close executed",
                            symbol=context.symbol,
                            side="SELL" if side == "BUY" else "BUY",
                        )
                    except Exception as close_err:
                        logger.critical(
                            "EMERGENCY_CLOSE_FAILED: Position remains on exchange "
                            "without protection! Manual intervention required.",
                            symbol=context.symbol, error=str(close_err),
                        )
                    return
            else:
                protection_data = {
                    "stop_order_id": f"virtual_stop_{context.execution_id}",
                    "stop_client_order_id": f"SL_{context.execution_id}",
                    "tp_order_id": f"virtual_tp_{context.execution_id}",
                    "tp_client_order_id": f"TP_{context.execution_id}",
                    "stop_price": stop_loss,
                    "tp_price": take_profit,
                }

            order_id = auth.get("orderId", "")
            payload = {
                "type": "entry",
                "execution_mode": context.execution_mode,
                "origin": context.origin,
                "protection_orders": protection_data,
                "order_id": order_id,
                "symbol": context.symbol,
                "side": trade_side,
                "avg_price": avg_price,
                "executed_qty": executed_qty,
                "commission": commission,
                "anchor_symbol": context.anchor_symbol,
                "correlation_score": context.correlation_score,
                "initial_stop_loss": stop_loss,
                "initial_take_profit": take_profit,
                "entry_thesis": context.entry_thesis,
                "execution_id": context.execution_id,
                "trade_group_id": context.trade_group_id,
                "candidate_id": context.candidate_id,
                "correlation_id": context.correlation_id,
                "llm_request_id": context.llm_request_id,
                "strategy_version": context.strategy_version,
                "execution_model": context.execution_model,
                "execution_model_version": context.execution_model_version,
                "execution_parameters": dict(context.execution_parameters),
                "risk_decision": context.risk_decision,
                "risk_decision_reason": context.risk_decision_reason,
                "created_by": "SCANNER",
                "opportunity_source": "SCANNER",
                "entry_timestamp": context.entry_timestamp.isoformat() if hasattr(context.entry_timestamp, 'isoformat') else str(context.entry_timestamp),
            }

            if context.execution_mode != "LIVE":
                payload["virtual_fill"] = VirtualFill(
                    avg_price=auth["avgPrice"],
                    executed_qty=auth["executedQty"],
                    fees=auth.get("commission", 0.0),
                    slippage_bps=auth.get("synthetic_slippage_bps", 0.0),
                    spread_bps=auth.get("synthetic_spread_bps", 0.0),
                    fee_bps=auth.get("synthetic_fee_bps", 0.0),
                ).model_dump()

            event = SystemEvent(
                event_type="ORDER_FILLED",
                service_name="ExecutionService",
                payload=payload,
            )
            await self._event_bus.publish(event)
            logger.info(
                "Entry order executed",
                symbol=context.symbol,
                side=trade_side,
                qty=executed_qty,
                price=avg_price,
                execution_mode=context.execution_mode,
                origin=context.origin,
                execution_id=context.execution_id,
                trade_group_id=context.trade_group_id,
            )

            if context.execution_mode == "LIVE" and self._mirror_enabled:
                await self._create_mirror(context, avg_price, stop_loss, take_profit, executed_qty)

        except InsufficientMarginError as e:
            logger.warning(
                "Insufficient margin — entry skipped",
                symbol=context.symbol,
                execution_mode=context.execution_mode,
                error=str(e),
            )
        except Exception as e:
            logger.error(
                "Entry execution failed",
                symbol=context.symbol,
                execution_mode=context.execution_mode,
                error=str(e),
            )

    async def _create_mirror(
        self,
        source_context: ExecutionContext,
        live_avg_price: float,
        stop_loss: float,
        take_profit: float,
        executed_qty: float,
    ) -> None:
        try:
            mirror_context = ExecutionContext(
                correlation_id=source_context.correlation_id,
                execution_id=str(uuid.uuid4()),
                trade_group_id=source_context.trade_group_id,
                candidate_id=source_context.candidate_id,
                strategy_version=source_context.strategy_version,
                llm_request_id=source_context.llm_request_id,
                execution_mode="SHADOW",
                origin="MIRROR",
                symbol=source_context.symbol,
                side=source_context.side,
                quantity=source_context.quantity,
                anchor_symbol=source_context.anchor_symbol,
                correlation_score=source_context.correlation_score,
                entry_thesis=source_context.entry_thesis,
                llm_confidence=source_context.llm_confidence,
                risk_decision=source_context.risk_decision,
                risk_decision_reason=source_context.risk_decision_reason,
                execution_model=source_context.execution_model,
                execution_model_version=source_context.execution_model_version,
                execution_parameters=dict(source_context.execution_parameters),
                entry_timestamp=source_context.entry_timestamp,
            )

            qty = executed_qty
            qty_str = str(qty)
            if qty <= 0:
                logger.error("Mirror quantity invalid", symbol=source_context.symbol)
                return

            mirror_side = source_context.side
            mirror_auth = await self._virtual.execute_entry(
                source_context.symbol, mirror_side, qty_str, mirror_context
            )

            mirror_price = mirror_auth["avgPrice"]
            mirror_commission = mirror_auth.get("commission", 0.0)

            mirror_protection = {
                "stop_order_id": f"virtual_stop_{mirror_context.execution_id}",
                "stop_client_order_id": f"SL_{mirror_context.execution_id}",
                "tp_order_id": f"virtual_tp_{mirror_context.execution_id}",
                "tp_client_order_id": f"TP_{mirror_context.execution_id}",
                "stop_price": stop_loss,
                "tp_price": take_profit,
            }
            mirror_payload = {
                "type": "entry",
                "execution_mode": "SHADOW",
                "origin": "MIRROR",
                "protection_orders": mirror_protection,
                "order_id": mirror_auth.get("orderId", ""),
                "symbol": source_context.symbol,
                "side": "LONG" if mirror_side == "BUY" else "SHORT",
                "avg_price": mirror_price,
                "executed_qty": qty,
                "commission": mirror_commission,
                "anchor_symbol": source_context.anchor_symbol,
                "correlation_score": source_context.correlation_score,
                "initial_stop_loss": stop_loss,
                "initial_take_profit": take_profit,
                "entry_thesis": source_context.entry_thesis,
                "execution_id": mirror_context.execution_id,
                "trade_group_id": source_context.trade_group_id,
                "candidate_id": source_context.candidate_id,
                "correlation_id": source_context.correlation_id,
                "llm_request_id": source_context.llm_request_id,
                "strategy_version": source_context.strategy_version,
                "execution_model": source_context.execution_model,
                "execution_model_version": source_context.execution_model_version,
                "execution_parameters": dict(source_context.execution_parameters),
                "risk_decision": source_context.risk_decision,
                "risk_decision_reason": source_context.risk_decision_reason,
                "created_by": "SCANNER",
                "opportunity_source": "SCANNER",
                "mirror_position_id": None,
                "entry_timestamp": source_context.entry_timestamp.isoformat() if hasattr(source_context.entry_timestamp, 'isoformat') else str(source_context.entry_timestamp),
            }

            mirror_payload["virtual_fill"] = VirtualFill(
                avg_price=mirror_price,
                executed_qty=qty,
                fees=mirror_commission,
                slippage_bps=mirror_auth.get("synthetic_slippage_bps", 0.0),
                spread_bps=mirror_auth.get("synthetic_spread_bps", 0.0),
                fee_bps=mirror_auth.get("synthetic_fee_bps", 0.0),
            ).model_dump()

            mirror_event = SystemEvent(
                event_type="ORDER_FILLED",
                service_name="ExecutionService",
                payload=mirror_payload,
            )
            await self._event_bus.publish(mirror_event)
            logger.info(
                "Mirror position created",
                trade_group_id=source_context.trade_group_id,
                symbol=source_context.symbol,
                mirror_execution_id=mirror_context.execution_id,
            )

        except Exception as e:
            logger.error(
                "Mirror creation failed",
                trade_group_id=source_context.trade_group_id,
                symbol=source_context.symbol,
                error=str(e),
            )

    async def execute_exit(self, position: Position, reason: str) -> None:
        try:
            side = "SELL" if position.side == "LONG" else "BUY"

            if position.execution_mode == "LIVE":
                qty_str = await self._live.round_quantity(position.symbol, position.quantity)
            else:
                qty_str = f"{round(position.quantity, 4):.4f}"

            if float(qty_str) <= 0:
                logger.error(
                    "Exit quantity invalid after rounding",
                    symbol=position.symbol,
                    raw_qty=position.quantity,
                )
                return

            if position.execution_mode == "LIVE":
                await self._live.cancel_protection(
                    position.symbol,
                    stop_client_order_id=position.protection_orders.stop_client_order_id,
                    tp_client_order_id=position.protection_orders.tp_client_order_id,
                )
                logger.info(
                    "Protection cancelled before exit",
                    symbol=position.symbol,
                    position_id=position.position_id,
                )

                auth = await self._live.execute_exit(
                    position.symbol, side, qty_str, position
                )
            else:
                auth = await self._virtual.execute_exit(
                    position.symbol, side, qty_str, position
                )

            avg_price = auth["avgPrice"]
            commission = auth.get("commission", 0.0)

            payload = {
                "type": "exit",
                "order_id": auth.get("orderId"),
                "position_id": position.position_id,
                "symbol": position.symbol,
                "side": position.side,
                "exit_price": avg_price,
                "commission": commission,
                "reason": reason,
                "execution_mode": position.execution_mode,
                "origin": position.origin,
                "trade_group_id": position.trade_group_id,
                "execution_id": position.execution_id,
            }

            if position.execution_mode != "LIVE":
                payload["virtual_fill"] = VirtualFill(
                    avg_price=auth["avgPrice"],
                    executed_qty=float(qty_str),
                    fees=auth.get("commission", 0.0),
                    slippage_bps=auth.get("synthetic_slippage_bps", 0.0),
                    spread_bps=auth.get("synthetic_spread_bps", 0.0),
                    fee_bps=auth.get("synthetic_fee_bps", 0.0),
                ).model_dump()

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
                execution_mode=position.execution_mode,
                source="fill_response",
                order_id=auth.get("orderId"),
                filled_qty=auth.get("executedQty"),
                commission=auth.get("commission", 0.0),
            )

        except Exception as e:
            logger.error(
                "Exit execution failed",
                symbol=position.symbol,
                error=str(e),
            )

    async def update_trailing_stop(self, position: Position, new_stop_price: float) -> dict:
        if position.execution_mode == "LIVE":
            hard_stop_side = "SELL" if position.side == "LONG" else "BUY"
            result = await self._live.update_trailing_stop(
                symbol=position.symbol,
                side=hard_stop_side,
                new_stop_price=new_stop_price,
                position_id=position.position_id,
                old_stop_client_order_id=position.protection_orders.stop_client_order_id,
                quantity=position.quantity,
                current_price=position.avg_fill_price,
            )
            position.protection_orders.stop_price = new_stop_price
            position.protection_orders.stop_order_id = str(result.get("order_id", ""))
            position.protection_orders.stop_client_order_id = result.get("client_order_id", "")
            position.protection_orders.last_updated = datetime.utcnow()
            position.protection_orders.status = "UPDATED"
        else:
            position.protection_orders.stop_price = new_stop_price
            position.protection_orders.stop_order_id = f"virtual_stop_update_{int(time.time())}"
            position.protection_orders.last_updated = datetime.utcnow()
            position.protection_orders.status = "UPDATED"
            result = {
                "order_id": position.protection_orders.stop_order_id,
                "client_order_id": position.protection_orders.stop_client_order_id,
                "stop_price": new_stop_price,
            }

        position.current_stop = new_stop_price
        logger.info(
            "Trailing stop updated",
            position_id=position.position_id,
            symbol=position.symbol,
            new_stop=new_stop_price,
            execution_mode=position.execution_mode,
        )
        return result
