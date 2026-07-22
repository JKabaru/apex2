from __future__ import annotations

import asyncio
import time
import uuid
from time import perf_counter
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime
from typing import Any

import aiohttp
import structlog

from src.api.binance_client import BinanceClient, BinanceClientError
from src.core.events import EventBus
from src.core.models import ExecutionContext, Position, PositionState, ProtectionOrders, SystemEvent, VirtualFill
from src.services.market_context import MarketContextService
from src.services.portfolio_manager import PortfolioManager
from src.models.execution import (
    ExecutableTrade,
    ExecutionPlan,
    ExecutedTrade,
    ExecutionStatus,
    TradeValidationReport,
    ValidationOutcomeStatus,
)
from src.services.validator import ConsistencyValidator, IntentValidator, ExchangeValidator

logger = structlog.get_logger("execution_service")

MAX_FILL_POLL_RETRIES = 5


def _log_execution_task_failure(
    task: asyncio.Task, symbol: str, execution_id: str,
) -> None:
    try:
        exc = task.exception()
        if exc is not None:
            logger.critical(
                "EXECUTE_ENTRY_TASK_CRASHED — fire-and-forget task failed silently",
                symbol=symbol,
                execution_id=execution_id,
                error=str(exc),
                exc_info=exc,
            )
    except asyncio.CancelledError:
        logger.warning(
            "EXECUTE_ENTRY_TASK_CANCELLED",
            symbol=symbol,
            execution_id=execution_id,
        )
    except Exception as e:
        logger.error(
            "EXECUTE_TASK_CALLBACK_ERROR",
            symbol=symbol,
            execution_id=execution_id,
            error=str(e),
            exc_info=True,
        )


def safe_float(value, default: float = 0.0) -> float:
    """Convert *value* to float, returning *default* for None, NaN, or bad types."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    return result


class InsufficientMarginError(Exception):
    pass


class CriticalProtectionFailure(Exception):
    pass


class BreakerTripped(Exception):
    pass


FILL_POLL_INTERVAL = 1


class IExecutor(ABC):
    @abstractmethod
    async def execute_entry(
        self, symbol: str, side: str, quantity_str: str, context: ExecutionContext,
        current_price: float = None,
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

    @property
    def client(self) -> BinanceClient:
        return self._client

    async def round_quantity(self, symbol: str, raw_qty: float, round_up: bool = False) -> str:
        step_size = await self._client.get_symbol_step_size(symbol)
        qty = Decimal(str(raw_qty))
        step = Decimal(str(step_size))
        precision = abs(step.as_tuple().exponent)
        if round_up:
            valid_qty = (qty / step).quantize(Decimal('1'), rounding=ROUND_UP) * step
        else:
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

        # ── Inline fill recovery: one last check, then fall back to position query ──
        logger.warning(
            "FILL_POLL_EXHAUSTED — attempting inline recovery",
            symbol=symbol,
            client_order_id=client_order_id,
            attempts=MAX_FILL_POLL_RETRIES,
        )
        try:
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
                    "INLINE_RECOVERY — order found FILLED on final check",
                    symbol=symbol,
                    order_id=order.get("orderId"),
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
        except Exception as e:
            logger.warning(
                "INLINE_RECOVERY — final order status check failed",
                symbol=symbol,
                error=str(e),
            )

        # Fall back to exchange position query
        try:
            exchange_positions = await self._client.get_open_positions()
            for ex_pos in exchange_positions:
                if ex_pos.get("symbol") == symbol:
                    entry_price = ex_pos.get("entry_price")
                    position_amt = ex_pos.get("position_amt", "0")
                    if entry_price and float(entry_price) > 0 and float(position_amt) != 0:
                        logger.info(
                            "INLINE_RECOVERY — position found on exchange via fallback",
                            symbol=symbol,
                            entry_price=entry_price,
                            position_amt=position_amt,
                        )
                        return {
                            "avgPrice": float(entry_price),
                            "executedQty": abs(float(position_amt)),
                            "cumQuote": abs(float(position_amt)) * float(entry_price),
                            "commission": 0.0,
                            "fills": [],
                            "status": "FILLED",
                            "orderId": None,
                        }
        except Exception as e:
            logger.warning(
                "INLINE_RECOVERY — position query failed",
                symbol=symbol,
                error=str(e),
            )

        raise BinanceClientError(
            f"Order {client_order_id} for {symbol} not FILLED after "
            f"{MAX_FILL_POLL_RETRIES} retries and inline recovery"
        )

    async def execute_entry(
        self, symbol: str, side: str, quantity_str: str, context: ExecutionContext,
        current_price: float = None,
    ) -> dict:
        position_side = "BOTH"

        clean_tgid = context.trade_group_id.replace("-", "")
        client_order_id = f"ex_{symbol}_{clean_tgid}"[:36]

        leverage = int(self._config.get("execution", {}).get("leverage", 10))
        logger.info(
            "LiveExecutor.execute_entry: setting leverage",
            symbol=symbol, leverage=leverage,
        )
        await self._client.set_leverage(symbol, leverage)

        if not current_price or current_price <= 0:
            raise BinanceClientError(
                f"Cannot execute entry for {symbol}: "
                f"invalid current_price={current_price}"
            )

        usdt_notional = float(quantity_str)
        base_qty = usdt_notional / current_price
        base_qty_str = await self.round_quantity(symbol, base_qty, round_up=False)
        logger.info(
            "LiveExecutor.execute_entry: converted USDT notional to base quantity",
            symbol=symbol,
            usdt_notional=usdt_notional,
            current_price=current_price,
            base_qty=base_qty,
            rounded_base_qty=base_qty_str,
        )

        # ── Outbound circuit breaker ──
        exec_cfg = self._config.get("execution", {})
        sizing_value = float(exec_cfg.get("sizing_value", 2.0))
        hard_limit = sizing_value * leverage * 1.2
        await self._check_exposure_breaker(symbol, usdt_notional, hard_limit)

        logger.info(
            "LiveExecutor.execute_entry: placing market order",
            symbol=symbol,
            side=side,
            position_side=position_side,
            quantity=base_qty_str,
            client_order_id=client_order_id,
        )
        try:
            post_result = await self._client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=base_qty_str,
                position_side=position_side,
                new_client_order_id=client_order_id,
            )
        except BinanceClientError as e:
            logger.critical(
                "BINANCE_API_REJECTED_ORDER",
                symbol=symbol,
                side=side,
                position_side=position_side,
                quantity=quantity_str,
                error=str(e),
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.critical(
                "BINANCE_API_UNEXPECTED_ERROR",
                symbol=symbol,
                side=side,
                quantity=quantity_str,
                error=str(e),
                exc_info=True,
            )
            raise

        order_id = post_result.get("orderId")
        returned_client_id = post_result.get("clientOrderId", client_order_id)
        poll_id = returned_client_id if returned_client_id else client_order_id

        fill = await self._poll_authoritative_fill(symbol, poll_id)

        # Post-execution position verification
        await self._verify_position_open(symbol, client_order_id, fill.get("executedQty", 0))

        return fill

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

    @staticmethod
    def _protection_id(value) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _verify_algo_order(
        order: dict,
        expected_side: str,
        expected_trigger: float,
        expected_type: str,
    ) -> bool:
        if order.get("side") != expected_side:
            return False
        order_type = order.get("type") or order.get("orderType") or ""
        if order_type != expected_type:
            return False
        actual_price = float(order.get("triggerPrice", 0.0))
        if abs(actual_price - expected_trigger) / max(expected_trigger, 0.01) > 0.001:
            return False
        return True

    async def _verify_protection_on_exchange(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        tp_price: float,
        stop_client_id: str,
        tp_client_id: str,
        require_tp: bool = True,
    ) -> bool:
        try:
            algo_orders = await self._client.get_open_algo_orders(symbol)
        except Exception as e:
            logger.warning(
                "Failed to query algo orders for verification",
                symbol=symbol, error=str(e),
            )
            return False

        stop_type = "STOP_MARKET"
        tp_type = "TAKE_PROFIT_MARKET"
        # side is the close order side (SELL for long, BUY for short) for both SL and TP
        close_side = side

        stop_ok = tp_ok = not require_tp
        for o in algo_orders:
            cid = o.get("clientAlgoId", "")
            order_type = o.get("type") or o.get("orderType") or ""
            order_side = o.get("side", "")
            if order_type == stop_type and order_side == close_side:
                if cid == stop_client_id or not stop_ok:
                    stop_ok = self._verify_algo_order(o, close_side, stop_price, stop_type)
            elif require_tp and order_type == tp_type and order_side == close_side:
                if cid == tp_client_id or not tp_ok:
                    tp_ok = self._verify_algo_order(o, close_side, tp_price, tp_type)

        return stop_ok and tp_ok

    @staticmethod
    def _is_duplicate_protection_error(error: Exception) -> bool:
        return "-4130" in str(error)

    async def _link_existing_stop(
        self, symbol: str, side: str, stop_client_id: str,
    ) -> dict | None:
        try:
            algo_orders = await self._client.get_open_algo_orders(symbol)
        except Exception:
            return None
        existing = self._find_existing_stop(algo_orders, side)
        if existing is None:
            return None
        logger.info(
            "Linked to existing exchange stop",
            symbol=symbol,
            client_algo_id=existing.get("clientAlgoId"),
            algo_id=existing.get("algoId"),
        )
        return {
            "algoId": existing.get("algoId"),
            "clientAlgoId": existing.get("clientAlgoId", stop_client_id),
            "triggerPrice": existing.get("triggerPrice"),
        }

    async def place_protection(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        tp_price: float,
        position_id: str,
        quantity: float,
        current_price: float,
        max_retries: int = None,
    ) -> dict:
        import math
        if not (stop_price > 0 and math.isfinite(stop_price)):
            raise CriticalProtectionFailure(
                f"Invalid stop_price={stop_price} for {symbol} position {position_id}"
            )
        if not (tp_price > 0 and math.isfinite(tp_price)):
            raise CriticalProtectionFailure(
                f"Invalid tp_price={tp_price} for {symbol} position {position_id}"
            )
        if not (quantity > 0 and math.isfinite(quantity)):
            raise CriticalProtectionFailure(
                f"Invalid quantity={quantity} for {symbol} position {position_id}"
            )

        if max_retries is None:
            max_retries = int(self._config.get("protection", {}).get("max_retry_attempts", 3))

        sid = self._short_id(position_id)
        stop_client_id = f"SL_{sid}"
        tp_client_id = f"TP_{sid}"
        last_error = None
        stop_data = await self._link_existing_stop(symbol, side, stop_client_id)

        if stop_data is None:
            for attempt in range(max_retries):
                try:
                    stop_data = await self._client.place_algo_stop(
                        symbol, side, stop_price, position_id,
                        client_algo_id=stop_client_id,
                        estimated_qty=quantity, current_price=current_price,
                    )
                    last_error = None
                    break
                except BinanceClientError as e:
                    if self._is_duplicate_protection_error(e):
                        stop_data = await self._link_existing_stop(symbol, side, stop_client_id)
                        if stop_data is not None:
                            last_error = None
                            break
                    if "-2021" in str(e):
                        try:
                            fresh_mark = await self._client.get_mark_price(symbol)
                            safe_stop = fresh_mark * (0.99 if side == "SELL" else 1.01)
                            corrected_id = f"SL_{sid}_s{attempt}"
                            stop_data = await self._client.place_algo_stop(
                                symbol, side, safe_stop, position_id,
                                client_algo_id=corrected_id,
                                estimated_qty=quantity, current_price=fresh_mark,
                            )
                            stop_client_id = corrected_id
                            stop_price = safe_stop
                            last_error = None
                            logger.info(
                                "SL_SAFETY_PLACED",
                                symbol=symbol,
                                original_stop=stop_price,
                                safe_stop=safe_stop,
                                fresh_mark=fresh_mark,
                            )
                            break
                        except Exception as e2:
                            logger.warning(
                                "SL_SAFETY_RETRY_FAILED",
                                symbol=symbol,
                                original_stop=stop_price,
                                error=str(e2),
                            )
                            last_error = e
                    elif self._is_permanent_failure(e):
                        raise CriticalProtectionFailure(
                            f"Permanent stop placement failure for {symbol} "
                            f"position {position_id}: {e}"
                        )
                    else:
                        logger.warning(
                            "Stop placement attempt failed",
                            symbol=symbol, attempt=attempt + 1, error=str(e),
                        )
                        last_error = e
                    await asyncio.sleep(1 * (attempt + 1))
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    logger.warning(
                        "Stop placement network error, retrying",
                        symbol=symbol, attempt=attempt + 1, error=str(e),
                    )
                    await asyncio.sleep(1 * (attempt + 1))

        if stop_data is None and last_error is not None:
            raise CriticalProtectionFailure(
                f"Stop placement failed after {max_retries} retries "
                f"for {symbol} position {position_id}: {last_error}"
            )

        linked_stop_cid = stop_data.get("clientAlgoId", stop_client_id)
        if linked_stop_cid:
            stop_client_id = linked_stop_cid

        tp_data = None
        tp_skipped = False
        for attempt in range(max_retries):
            try:
                tp_data = await self._client.place_algo_tp(
                    symbol, side, tp_price, position_id,
                    client_algo_id=tp_client_id,
                    estimated_qty=quantity, current_price=current_price,
                )
                last_error = None
                break
            except BinanceClientError as e:
                if self._is_duplicate_protection_error(e):
                    try:
                        algo_orders = await self._client.get_open_algo_orders(symbol)
                        existing_tp = next(
                            (
                                o for o in algo_orders
                                if (o.get("type") or o.get("orderType")) == "TAKE_PROFIT_MARKET"
                                and o.get("side") == side
                            ),
                            None,
                        )
                        if existing_tp is not None:
                            tp_data = existing_tp
                            tp_client_id = existing_tp.get("clientAlgoId", tp_client_id)
                            last_error = None
                            break
                    except Exception:
                        pass
                if "-2021" in str(e):
                    logger.warning(
                        "TP would trigger immediately — continuing with stop-only protection",
                        symbol=symbol, tp_price=tp_price, current_price=current_price,
                    )
                    tp_skipped = True
                    last_error = None
                    break
                if self._is_permanent_failure(e):
                    logger.warning(
                        "Permanent TP failure — continuing with stop-only protection",
                        symbol=symbol, error=str(e),
                    )
                    tp_skipped = True
                    last_error = None
                    break
                logger.warning(
                    "TP placement attempt failed",
                    symbol=symbol, attempt=attempt + 1, error=str(e),
                )
                last_error = e
                await asyncio.sleep(1 * (attempt + 1))
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(
                    "TP placement network error, retrying",
                    symbol=symbol, attempt=attempt + 1, error=str(e),
                )
                await asyncio.sleep(1 * (attempt + 1))

        if last_error is not None and not tp_skipped:
            logger.warning(
                "TP placement failed — keeping stop-only protection",
                symbol=symbol, position_id=position_id, error=str(last_error),
            )
            tp_skipped = True

        result = {
            "stop_order_id": str(stop_data.get("algoId", "")),
            "stop_client_order_id": stop_client_id,
            "tp_order_id": str(tp_data.get("algoId", "")) if tp_data else None,
            "tp_client_order_id": tp_client_id if tp_data else None,
            "stop_price": stop_price,
            "tp_price": tp_price if tp_data else 0.0,
            "status": "ACTIVE" if not tp_skipped else "STOP_ONLY",
        }

        verified = await self._verify_protection_on_exchange(
            symbol, side, stop_price, tp_price, stop_client_id, tp_client_id,
            require_tp=not tp_skipped,
        )
        if not verified:
            # Lenient fallback: any close-side stop on exchange counts as protected
            try:
                algo_orders = await self._client.get_open_algo_orders(symbol)
                if self._find_existing_stop(algo_orders, side) is not None:
                    logger.warning(
                        "PROTECTION_VERIFICATION_LENIENT_PASS: stop present on exchange",
                        symbol=symbol, position_id=position_id,
                    )
                    result["status"] = "ACTIVE" if not tp_skipped else "STOP_ONLY"
                    return result
            except Exception:
                pass
            logger.error(
                "PROTECTION_VERIFICATION_FAILED",
                symbol=symbol, position_id=position_id,
            )
            raise CriticalProtectionFailure(
                f"Protection verification failed for {symbol} position {position_id}"
            )

        logger.info(
            "PROTECTION_VERIFIED",
            symbol=symbol, position_id=position_id,
            stop_order_id=stop_data.get("algoId"),
            tp_order_id=tp_data.get("algoId") if tp_data else None,
            stop_only=tp_skipped,
        )
        return result

    @staticmethod
    def _is_permanent_failure(error: Exception) -> bool:
        if isinstance(error, BinanceClientError):
            error_str = str(error).lower()
            permanent_indicators = [
                "invalid", "-2010", "-2011", "-1013", "-1111",
                "precision", "filter", "notional", "minimum",
                "reduce only", "position side",
            ]
            return any(indicator in error_str for indicator in permanent_indicators)
        return False

    @staticmethod
    def _is_stop_market_order(order: dict, side: str) -> bool:
        order_type = order.get("type") or order.get("orderType") or ""
        return order_type == "STOP_MARKET" and order.get("side") == side

    @staticmethod
    def _is_tp_market_order(order: dict, side: str) -> bool:
        order_type = order.get("type") or order.get("orderType") or ""
        return order_type == "TAKE_PROFIT_MARKET" and order.get("side") == side

    @staticmethod
    def _find_existing_stop(open_orders: list[dict], side: str) -> dict | None:
        return next(
            (o for o in open_orders if LiveExecutor._is_stop_market_order(o, side)),
            None,
        )

    @staticmethod
    def _find_existing_tp(open_orders: list[dict], side: str) -> dict | None:
        return next(
            (o for o in open_orders if LiveExecutor._is_tp_market_order(o, side)),
            None,
        )

    async def _cancel_existing_stop(self, symbol: str, existing_stop: dict) -> bool:
        """Cancel an existing stop order. Returns True if cleared or already gone."""
        client_algo_id = existing_stop.get("clientAlgoId")
        algo_id = existing_stop.get("algoId")
        if not client_algo_id and algo_id is None:
            logger.error(
                "Existing stop found but has no cancel identifier",
                symbol=symbol,
                order=existing_stop,
            )
            return False

        try:
            await self._client.cancel_algo_order(
                symbol,
                client_algo_id=client_algo_id,
                algo_id=algo_id,
            )
            logger.info(
                "Cancelled old trailing stop",
                symbol=symbol,
                client_algo_id=client_algo_id,
                algo_id=algo_id,
            )
            return True
        except Exception as e:
            if "-4003" in str(e):
                logger.info(
                    "Old trailing stop already triggered or not found on exchange (-4003)",
                    symbol=symbol,
                    client_algo_id=client_algo_id,
                    algo_id=algo_id,
                )
                return True
            logger.error(
                "Failed to cancel old trailing stop",
                symbol=symbol,
                client_algo_id=client_algo_id,
                algo_id=algo_id,
                error=str(e),
            )
            return False

    async def _confirm_stop_cleared(self, symbol: str, side: str) -> bool:
        """Re-query exchange to confirm no STOP_MARKET remains for this side."""
        try:
            open_orders = await self._client.get_open_algo_orders(symbol)
        except Exception as e:
            logger.error(
                "Failed to confirm stop cancellation",
                symbol=symbol,
                error=str(e),
            )
            return False

        remaining = self._find_existing_stop(open_orders, side)
        if remaining:
            logger.error(
                "Stop still open after cancel attempt",
                symbol=symbol,
                side=side,
                client_algo_id=remaining.get("clientAlgoId"),
                algo_id=remaining.get("algoId"),
            )
            return False
        return True

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
        import math
        if not (new_stop_price > 0 and math.isfinite(new_stop_price)):
            logger.error(
                "TRAILING_STOP_INVALID_PRICE",
                symbol=symbol, new_stop_price=new_stop_price,
            )
            return {}

        sid = self._short_id(position_id)
        ts = int(time.time() * 1000)
        new_client_id = f"ST_{sid}_{ts}"

        # Atomic lifecycle under per-symbol lock prevents -4130 races
        async with self._client.algo_lock(symbol):
            logger.info(
                "Trailing stop update",
                symbol=symbol, old_stop_client_id=old_stop_client_order_id,
            )

            # 1. VERIFY EXISTING — check old stop is present on exchange
            if old_stop_client_order_id:
                try:
                    algo_orders = await self._client.get_open_algo_orders(symbol)
                    old_still_active = any(
                        o.get("clientAlgoId") == old_stop_client_order_id
                        for o in algo_orders
                    )
                    if not old_still_active:
                        logger.warning(
                            "Existing trailing stop not found on exchange",
                            symbol=symbol, old_client_id=old_stop_client_order_id,
                        )
                except Exception as e:
                    logger.error(
                        "TRAILING_STOP_VERIFY_EXISTING_FAILED",
                        symbol=symbol, error=str(e),
                    )

            # 2. CANCEL old trailing stop only (TP persists)
            if old_stop_client_order_id:
                try:
                    await self._client.cancel_algo_by_client_id(symbol, old_stop_client_order_id)
                except BinanceClientError as e:
                    err_str = str(e)
                    if "-2011" in err_str or "-4003" in err_str:
                        logger.info(
                            "Old trailing stop already triggered or not found on exchange",
                            symbol=symbol, old_client_id=old_stop_client_order_id, error=err_str,
                        )
                    else:
                        raise

            # 3. VERIFY CANCELLATION
            try:
                await asyncio.sleep(0.3)
                remaining = await self._client.get_open_algo_orders(symbol)
                if old_stop_client_order_id and any(
                    o.get("clientAlgoId") == old_stop_client_order_id for o in remaining
                ):
                    logger.warning(
                        "TRAILING_STOP_CANCELLATION_NOT_CONFIRMED",
                        symbol=symbol, old_client_id=old_stop_client_order_id,
                    )
                    await asyncio.sleep(0.5)
                    remaining = await self._client.get_open_algo_orders(symbol)
                    if any(o.get("clientAlgoId") == old_stop_client_order_id for o in remaining):
                        logger.error(
                            "TRAILING_STOP_CANCELLATION_FAILED",
                            symbol=symbol, old_client_id=old_stop_client_order_id,
                        )
                        return {}
            except Exception as e:
                logger.error(
                    "TRAILING_STOP_VERIFY_CANCELLATION_FAILED",
                    symbol=symbol, error=str(e),
                )

            # 4. WAIT for exchange settlement
            await asyncio.sleep(0.5)

            # 5. CREATE replacement
            try:
                new_data = await self._client.place_algo_stop(
                    symbol, side, new_stop_price, position_id,
                    client_algo_id=new_client_id,
                    estimated_qty=quantity, current_price=current_price,
                )
                logger.info("Placed new trailing stop", symbol=symbol, new_stop=new_stop_price)
            except Exception as e:
                err_str = str(e)
                logger.error(
                    "TRAILING_STOP_CREATE_FAILED",
                    symbol=symbol, error=err_str,
                )
                if "-2021" in err_str:
                    try:
                        fresh_mark = await self._client.get_mark_price(symbol)
                    except Exception as mp_err:
                        logger.error(
                            "TRAILING_MARK_PRICE_FAILED",
                            symbol=symbol, error=str(mp_err),
                        )
                        return {"status": "price_invalid"}
                    safe_stop = fresh_mark * (0.99 if side == "SELL" else 1.01)
                    corrected_id = f"ST_{sid}_{ts}_r"
                    try:
                        new_data = await self._client.place_algo_stop(
                            symbol, side, safe_stop, position_id,
                            client_algo_id=corrected_id,
                            estimated_qty=quantity, current_price=fresh_mark,
                        )
                        original_price = new_stop_price
                        new_client_id = corrected_id
                        new_stop_price = safe_stop
                        logger.info(
                            "TRAILING_STOP_SAFETY_PLACED",
                            symbol=symbol,
                            original_price=original_price,
                            safe_stop=safe_stop,
                            fresh_mark=fresh_mark,
                        )
                    except Exception as e2:
                        logger.critical(
                            "TRAILING_STOP_SAFETY_FAILED",
                            symbol=symbol,
                            original_price=new_stop_price,
                            safe_stop=safe_stop,
                            fresh_mark=fresh_mark,
                            error=str(e2),
                        )
                        return {"status": "safety_failed", "error": str(e2)}
                elif "-4130" in err_str:
                    try:
                        remaining = await self._client.get_open_algo_orders(symbol)
                        existing = self._find_existing_stop(remaining, side)
                        if existing is not None:
                            logger.info(
                                "-4130: Linked to existing stop during trailing update",
                                symbol=symbol,
                                algo_id=existing.get("algoId"),
                                client_algo_id=existing.get("clientAlgoId"),
                                trigger_price=existing.get("triggerPrice"),
                            )
                            return {
                                "order_id": existing.get("algoId"),
                                "client_order_id": existing.get("clientAlgoId", new_client_id),
                                "stop_price": float(existing.get("triggerPrice", new_stop_price)),
                            }
                    except Exception:
                        pass
                    return {"status": "transient_failure", "error": err_str}
                else:
                    return {"status": "transient_failure", "error": err_str}

            # 6. VERIFY replacement exists on exchange
            try:
                await asyncio.sleep(0.3)
                algo_orders = await self._client.get_open_algo_orders(symbol)
                verified = any(o.get("clientAlgoId") == new_client_id for o in algo_orders)
                if not verified:
                    logger.error(
                        "TRAILING_STOP_VERIFICATION_FAILED",
                        symbol=symbol, new_client_id=new_client_id,
                    )
                    return {}
            except Exception as e:
                logger.error(
                    "TRAILING_STOP_VERIFICATION_ERROR",
                    symbol=symbol, error=str(e),
                )
                return {}

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

    async def _verify_position_open(
        self, symbol: str, client_order_id: str, expected_qty: float,
    ) -> None:
        try:
            positions = await self._client.get_open_positions()
            pos = next((p for p in positions if p.get("symbol") == symbol), None)
            if pos is None:
                logger.critical(
                    "POST_EXECUTION_VERIFICATION_FAILED: Position NOT found on exchange after fill",
                    symbol=symbol,
                    client_order_id=client_order_id,
                    expected_qty=expected_qty,
                    open_positions=[p.get("symbol") for p in positions],
                )
            else:
                pos_amt = float(pos.get("positionAmt", 0))
                logger.info(
                    "POST_EXECUTION_VERIFICATION_PASSED: Position confirmed on exchange",
                    symbol=symbol,
                    position_amt=pos_amt,
                    expected_qty=expected_qty,
                    unrealized_pnl=pos.get("unRealizedProfit"),
                )
        except Exception as e:
            logger.error(
                "POST_EXECUTION_VERIFICATION_ERROR",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )

    async def _check_exposure_breaker(self, symbol: str, new_notional: float, hard_limit: float) -> None:
        try:
            positions = await self._client.get_open_positions()
        except Exception as e:
            logger.error("Exposure breaker: failed to query positions, allowing order", symbol=symbol, error=str(e))
            return
        pos = next((p for p in positions if p.get("symbol") == symbol), None)
        current_notional = abs(pos.get("notional", 0.0)) if pos else 0.0

        try:
            orders = await self._client.get_open_orders(symbol)
        except Exception as e:
            logger.error("Exposure breaker: failed to query open orders, allowing order", symbol=symbol, error=str(e))
            return
        pending_notional = sum(
            float(o.get("origQty", 0)) * float(o.get("price", 0))
            for o in orders if o.get("side") == "BUY"
        )

        total = current_notional + pending_notional + new_notional
        if total > hard_limit:
            raise BreakerTripped(
                f"Exposure breaker tripped for {symbol}: {total:.4f} USDT exceeds "
                f"hard limit {hard_limit:.4f} USDT "
                f"(current={current_notional:.4f}, pending={pending_notional:.4f}, new={new_notional:.4f})"
            )
        logger.info(
            "Exposure breaker: safe",
            symbol=symbol,
            current=round(current_notional, 4),
            pending=round(pending_notional, 4),
            new=round(new_notional, 4),
            total=round(total, 4),
            limit=round(hard_limit, 4),
        )

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
                symbol=symbol, exc_info=True,
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
        self, symbol: str, side: str, quantity_str: str, context: ExecutionContext,
        current_price: float = None,
    ) -> dict:
        is_buy = side == "BUY"
        synthetic_price, slippage_bps, spread_bps, fee_bps = (
            await self._get_synthetic_price(symbol, is_buy, context)
        )
        notional = float(quantity_str)
        base_qty = notional / synthetic_price
        fee_cost = notional * fee_bps / 10000.0

        return {
            "avgPrice": synthetic_price,
            "executedQty": base_qty,
            "cumQuote": notional,
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
        self._protection_retry_count: dict[str, int] = {}
        self._trailing_failures: dict[str, int] = {}
        self._repair_in_progress: set[str] = set()  # dedup lock for concurrent repair requests
        self._event_bus.subscribe("EXECUTE_TRADE", self._on_execute_trade)
        self._event_bus.subscribe("PROTECTION_REPAIR_REQUESTED", self._on_protection_repair_requested)
        logger.info("ExecutionService initialized", mirror_enabled=mirror_enabled)

    async def _on_execute_trade(self, event: SystemEvent) -> None:
        payload = event.payload
        context_data = payload.get("context", {})
        context = ExecutionContext(**context_data)
        t_entry = perf_counter()
        logger.info(
            "EXECUTE_TRADE_RECEIVED",
            symbol=context.symbol,
            execution_mode=context.execution_mode,
            origin=context.origin,
            execution_id=context.execution_id,
            trade_group_id=context.trade_group_id,
            opportunity_id=context.opportunity_id,
            side=context.side,
            elapsed_ms=0,
        )
        task = asyncio.create_task(self.execute_entry(context))
        task.add_done_callback(
            lambda t: _log_execution_task_failure(t, context.symbol, context.execution_id)
        )

    async def _on_protection_repair_requested(self, event: SystemEvent) -> None:
        payload = event.payload
        symbol = payload.get("symbol")
        position_id = payload.get("position_id")
        if not symbol or not position_id:
            logger.error("PROTECTION_REPAIR_REQUESTED missing symbol or position_id", payload=payload)
            return

        # Dedup: if a repair is already in flight for this position, skip
        if position_id in self._repair_in_progress:
            logger.info(
                "PROTECTION_REPAIR_DEDUP_SKIPPED",
                position_id=position_id, symbol=symbol,
            )
            return

        self._repair_in_progress.add(position_id)
        try:
            position = self._portfolio.get_position_by_id(position_id)
            if position is None:
                logger.error("Position not found for protection repair", position_id=position_id)
                return
            if position.lifecycle_state in ("CLOSING", "CLOSED", "ARCHIVED"):
                logger.info("Position already terminal, skipping repair", position_id=position_id)
                return

            side = "SELL" if position.side == "LONG" else "BUY"
            stop_price = position.protection_orders.stop_price
            tp_price = position.protection_orders.tp_price
            qty = position.quantity
            price = position.avg_fill_price

            if qty <= 0 or price <= 0:
                logger.error(
                    "PROTECTION_REPAIR_INVALID_PARAMS",
                    symbol=symbol, position_id=position_id,
                    stop_price=stop_price, tp_price=tp_price,
                    qty=qty, price=price,
                )
                self._publish_audit_event("PROTECTION_REPAIR_FAILED", symbol, position_id, {
                    "reason": "invalid_params",
                    "stop_price": stop_price, "tp_price": tp_price,
                    "qty": qty, "price": price,
                })
                return

            if stop_price <= 0 or tp_price <= 0:
                exec_cfg = self._config.get("execution", {})
                stop_loss_pct = float(exec_cfg.get("stop_loss_pct", 0.98))
                take_profit_pct = float(exec_cfg.get("take_profit_pct", 1.04))
                if position.side == "LONG":
                    if stop_price <= 0:
                        stop_price = price * stop_loss_pct
                    if tp_price <= 0:
                        tp_price = price * take_profit_pct
                else:
                    if stop_price <= 0:
                        stop_price = price * (2.0 - stop_loss_pct)
                    if tp_price <= 0:
                        tp_price = price * (2.0 - take_profit_pct)
                position.protection_orders.stop_price = stop_price
                position.protection_orders.tp_price = tp_price
                logger.info(
                    "PROTECTION_REPAIR_PRICES_COMPUTED",
                    symbol=symbol, position_id=position_id,
                    stop_price=stop_price, tp_price=tp_price,
                )

            logger.info("PROTECTION_REPAIR_STARTED", symbol=symbol, position_id=position_id)
            self._publish_audit_event("PROTECTION_REPAIR_STARTED", symbol, position_id, {})

            stop_client_id = f"SL_{self._live._short_id(position_id)}"
            stop_max_retries = int(self._config.get("protection", {}).get("max_retry_attempts", 3))
            last_stop_error = None
            linked_existing = False
            for attempt in range(stop_max_retries):
                try:
                    stop_data = await self._live._client.place_algo_stop(
                        symbol, side, stop_price, position_id,
                        client_algo_id=stop_client_id,
                        estimated_qty=qty, current_price=price,
                    )
                    last_stop_error = None
                    break
                except BinanceClientError as e:
                    error_str = str(e)
                    if "-4130" in error_str:
                        try:
                            algo_orders = await self._live._client.get_open_algo_orders(symbol)
                            existing = self._live._find_existing_stop(algo_orders, side)
                            if existing is not None:
                                stop_data = existing
                                linked_existing = True
                                last_stop_error = None
                                logger.info(
                                    "-4130: Linked to existing stop on exchange",
                                    symbol=symbol, algo_id=existing.get("algoId"),
                                    client_algo_id=existing.get("clientAlgoId"),
                                )
                                break
                        except Exception:
                            pass
                        logger.warning(
                            "-4130: No matching existing stop found, will retry",
                            symbol=symbol, attempt=attempt + 1,
                        )
                    elif self._live._is_permanent_failure(e):
                        raise CriticalProtectionFailure(
                            f"Permanent stop repair failure for {symbol} position {position_id}: {e}"
                        )
                    logger.warning(
                        "Stop repair attempt failed",
                        symbol=symbol, attempt=attempt + 1, error=str(e),
                    )
                    last_stop_error = e
                    await asyncio.sleep(1 * (attempt + 1))
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_stop_error = e
                    logger.warning(
                        "Stop repair network error",
                        symbol=symbol, attempt=attempt + 1, error=str(e),
                    )
                    await asyncio.sleep(1 * (attempt + 1))

            if last_stop_error is not None:
                raise CriticalProtectionFailure(
                    f"Stop repair failed after {stop_max_retries} retries "
                    f"for {symbol} position {position_id}: {last_stop_error}"
                )

            await asyncio.sleep(0.3)
            algo_orders = await self._live._client.get_open_algo_orders(symbol)
            existing_stop = self._live._find_existing_stop(algo_orders, side)
            if existing_stop is None:
                raise CriticalProtectionFailure(
                    f"Stop repair verification failed for {symbol} position {position_id}"
                )
            self._protection_retry_count.pop(position_id, None)
            position.protection_orders.stop_order_id = str(existing_stop.get("algoId", "") or "")
            position.protection_orders.stop_client_order_id = existing_stop.get("clientAlgoId", stop_client_id)
            position.protection_orders.stop_price = float(existing_stop.get("triggerPrice", stop_price))

            tp_placed = False
            tp_client_id = f"TP_{self._live._short_id(position_id)}"
            if tp_price > 0:
                tp_max_retries = stop_max_retries
                last_tp_error = None
                for attempt in range(tp_max_retries):
                    try:
                        algo_orders = await self._live._client.get_open_algo_orders(symbol)
                        existing_tp = self._live._find_existing_tp(algo_orders, side)
                        if existing_tp is not None:
                            position.protection_orders.tp_order_id = self._live._protection_id(existing_tp.get("algoId"))
                            position.protection_orders.tp_client_order_id = existing_tp.get("clientAlgoId", tp_client_id)
                            position.protection_orders.tp_price = float(existing_tp.get("triggerPrice", tp_price))
                            tp_placed = True
                            last_tp_error = None
                            logger.info(
                                "Linked to existing TP on exchange",
                                symbol=symbol, algo_id=existing_tp.get("algoId"),
                            )
                            break
                    except Exception:
                        pass
                    try:
                        tp_data = await self._live._client.place_algo_tp(
                            symbol, side, tp_price, position_id,
                            client_algo_id=tp_client_id,
                            estimated_qty=qty, current_price=price,
                        )
                        last_tp_error = None
                        await asyncio.sleep(0.3)
                        algo_orders = await self._live._client.get_open_algo_orders(symbol)
                        if any(o.get("clientAlgoId") == tp_client_id for o in algo_orders):
                            position.protection_orders.tp_order_id = self._live._protection_id(tp_data.get("algoId"))
                            position.protection_orders.tp_client_order_id = tp_client_id
                            position.protection_orders.tp_price = tp_price
                            tp_placed = True
                            break
                    except BinanceClientError as e:
                        error_str = str(e)
                        if "-4130" in error_str:
                            try:
                                algo_orders = await self._live._client.get_open_algo_orders(symbol)
                                existing_tp = self._live._find_existing_tp(algo_orders, side)
                                if existing_tp is not None:
                                    position.protection_orders.tp_order_id = self._live._protection_id(existing_tp.get("algoId"))
                                    position.protection_orders.tp_client_order_id = existing_tp.get("clientAlgoId", tp_client_id)
                                    position.protection_orders.tp_price = float(existing_tp.get("triggerPrice", tp_price))
                                    tp_placed = True
                                    last_tp_error = None
                                    break
                            except Exception:
                                pass
                        elif "-2021" in error_str:
                            logger.warning(
                                "TP repair would trigger immediately — skipping TP",
                                symbol=symbol, tp_price=tp_price,
                            )
                            last_tp_error = None
                            break
                        elif self._live._is_permanent_failure(e):
                            logger.warning(
                                "Permanent TP failure during repair — continuing with stop-only",
                                symbol=symbol, error=str(e),
                            )
                            last_tp_error = None
                            break
                        logger.warning(
                            "TP repair attempt failed",
                            symbol=symbol, attempt=attempt + 1, error=str(e),
                        )
                        last_tp_error = e
                        await asyncio.sleep(1 * (attempt + 1))
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        last_tp_error = e
                        logger.warning(
                            "TP repair network error",
                            symbol=symbol, attempt=attempt + 1, error=str(e),
                        )
                        await asyncio.sleep(1 * (attempt + 1))
                if last_tp_error is not None:
                    logger.warning(
                        "TP repair failed after retries — continuing with stop-only protection",
                        symbol=symbol, position_id=position_id, error=str(last_tp_error),
                    )

            position.protection_orders.status = "ACTIVE" if position.lifecycle_state == PositionState.UNMANAGED_ADOPTED else "VERIFIED"
            position.protection_orders.last_updated = datetime.utcnow()
            if position.lifecycle_state == PositionState.UNMANAGED_ADOPTED:
                try:
                    await self._portfolio.update_position_state(
                        position.position_id, PositionState.OPEN,
                    )
                    logger.info(
                        "Adopted position promoted to OPEN after repair",
                        symbol=symbol, position_id=position.position_id,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to promote adopted position after repair",
                        symbol=symbol, position_id=position.position_id, error=str(e),
                    )
            self._publish_audit_event("PROTECTION_REPAIR_COMPLETED", symbol, position_id, {
                "stop_order_id": existing_stop.get("algoId"),
                "tp_order_id": position.protection_orders.tp_order_id if tp_placed else None,
                "tp_placed": tp_placed,
            })
        except CriticalProtectionFailure as e:
            error_str = str(e)
            if any(p in error_str.lower() for p in ["permanent", "invalid"]):
                self._publish_audit_event("PROTECTION_REPAIR_FAILED", symbol, position_id, {
                    "error": error_str, "reason": "permanent",
                })
                self._protection_retry_count.pop(position_id, None)
                return
            retry_count = self._protection_retry_count.get(position_id, 0) + 1
            max_retries = int(self._config.get("protection", {}).get("max_retry_attempts", 3))
            if retry_count >= max_retries:
                logger.critical(
                    "PROTECTION_REPAIR_EXHAUSTED",
                    symbol=symbol, position_id=position_id,
                    retry_count=retry_count, max_retries=max_retries,
                )
                self._publish_audit_event("PROTECTION_REPAIR_EXHAUSTED", symbol, position_id, {
                    "reason": "retry_exhausted", "retry_count": retry_count,
                })
                self._protection_retry_count.pop(position_id, None)
            else:
                self._protection_retry_count[position_id] = retry_count
                retry_interval = float(self._config.get("protection", {}).get("retry_interval_seconds", 5.0))
                logger.warning(
                    "PROTECTION_REPAIR_SCHEDULED_RETRY",
                    symbol=symbol, position_id=position_id,
                    attempt=retry_count, max_retries=max_retries,
                    next_retry_seconds=retry_interval,
                )
                self._publish_audit_event("PROTECTION_REPAIR_FAILED", symbol, position_id, {
                    "error": error_str, "reason": "retryable",
                    "attempt": retry_count, "max_retries": max_retries,
                })
                await asyncio.sleep(retry_interval)
                retry_payload = {
                    "position_id": position_id,
                    "symbol": symbol,
                    "side": position.side,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "quantity": qty,
                    "execution_id": position_id,
                }
                retry_event = SystemEvent(
                    event_type="PROTECTION_REPAIR_REQUESTED",
                    service_name="ExecutionService",
                    payload=retry_payload,
                )
                self._event_bus.publish_nowait(retry_event)
        finally:
            self._repair_in_progress.discard(position_id)

    async def _compute_sizing(
        self, context: ExecutionContext, current_price: float = None,
        exec_cfg: dict = None, available_balance: float = None,
    ) -> tuple[str, float]:
        if exec_cfg is None:
            exec_cfg = self._config.get("execution", {})
        sizing_mode = exec_cfg.get("sizing_mode", "fixed_usdt")
        sizing_value = float(exec_cfg.get("sizing_value", 2.0))
        leverage = int(exec_cfg.get("leverage", 10))
        max_risk_pct = float(exec_cfg.get("max_risk_pct", 0.02))
        stop_loss_pct = float(exec_cfg.get("stop_loss_pct", 0.98))
        stop_distance = abs(1.0 - stop_loss_pct)

        if sizing_mode == "fixed_usdt":
            if not (current_price and current_price > 0):
                logger.error(
                    "Cannot compute quantity: no price available for fixed_usdt sizing",
                    symbol=context.symbol, sizing_mode=sizing_mode,
                )
                return "0", 0.0
            raw_qty = sizing_value * leverage
            if stop_distance > max_risk_pct and max_risk_pct > 0:
                scale = max_risk_pct / stop_distance
                raw_qty = raw_qty * scale
                logger.info(
                    "Risk-based size scaling applied",
                    symbol=context.symbol,
                    scale=round(scale, 4),
                    stop_distance=round(stop_distance, 4),
                    max_risk_pct=max_risk_pct,
                    scaled_notional=round(raw_qty, 2),
                )
        elif sizing_mode == "risk_pct":
            if not (available_balance and available_balance > 0 and stop_distance > 0):
                logger.error(
                    "Cannot compute quantity for risk_pct sizing: invalid params",
                    symbol=context.symbol,
                    available_balance=available_balance,
                    stop_distance=stop_distance,
                )
                return "0", 0.0
            # Notional sized so max loss at stop ≈ balance × risk_pct.
            # Leverage affects margin, not loss for a given notional — do NOT multiply by leverage.
            raw_qty = (available_balance * max_risk_pct) / stop_distance
        else:
            logger.error(
                "Unknown sizing mode",
                symbol=context.symbol, sizing_mode=sizing_mode,
            )
            return "0", 0.0

        qty_str = f"{raw_qty:.2f}"
        qty = float(qty_str)
        max_allowed = float(exec_cfg.get("sizing_value", 2.0)) * int(exec_cfg.get("leverage", 10)) * 2
        if qty > max_allowed:
            logger.critical(
                "SIZING_CLAMP_TRIGGERED",
                symbol=context.symbol,
                raw_qty=raw_qty,
                clamped_to=max_allowed,
            )
            qty = max_allowed
            qty_str = f"{qty:.2f}"
        if qty <= 0:
            logger.error(
                "Position notional too small",
                symbol=context.symbol, raw_qty=raw_qty,
            )
            return "0", 0.0
        logger.info(
            "Computing position sizing",
            symbol=context.symbol,
            execution_mode=context.execution_mode,
            sizing_mode=sizing_mode,
            leverage=leverage,
            usdt_notional=qty,
            current_price=current_price,
        )
        return qty_str, qty

    def _build_execution_plan(
        self, context: ExecutionContext, exec_cfg: dict,
    ) -> ExecutionPlan:
        return ExecutionPlan(
            symbol=context.symbol,
            side=context.side,
            leverage=int(exec_cfg.get("leverage", 10)),
            expected_slippage_bps=float(exec_cfg.get("slippage_bps", 3.0)),
            execution_id=context.execution_id,
            trade_group_id=context.trade_group_id,
            opportunity_id=context.opportunity_id,
            llm_confidence=context.llm_confidence,
        )

    def _build_executable_trade(
        self,
        context: ExecutionContext,
        qty: float,
        qty_str: str,
        current_price: float,
        trade_side: str,
        proxy_stop: float,
        proxy_tp: float,
        exec_cfg: dict,
        exchange_filters: dict,
        taker_fee_rate: float,
        available_balance: float,
        indicators: dict,
        plan: ExecutionPlan,
    ) -> ExecutableTrade:
        sizing_value = float(exec_cfg.get("sizing_value", 2.0))
        stop_loss_pct = float(exec_cfg.get("stop_loss_pct", 0.98))
        take_profit_pct = float(exec_cfg.get("take_profit_pct", 1.04))
        max_risk_pct = float(exec_cfg.get("max_risk_pct", 0.02))
        slippage_bps = float(exec_cfg.get("slippage_bps", 3.0))

        expected_notional = qty
        expected_loss = qty * abs(1.0 - stop_loss_pct) if proxy_stop > 0 else 0.0
        expected_reward = qty * abs(take_profit_pct - 1.0) if proxy_tp > 0 else 0.0

        expected_entry_fee = expected_notional * taker_fee_rate
        expected_exit_fee = expected_notional * taker_fee_rate
        expected_slippage = expected_notional * (slippage_bps / 10000.0)

        price_loss = expected_loss
        worst_case_loss = (
            price_loss + expected_entry_fee + expected_exit_fee + expected_slippage
        )
        sizing_mode = exec_cfg.get("sizing_mode", "fixed_usdt")
        if sizing_mode == "risk_pct":
            max_allowed_risk = available_balance * max_risk_pct
        else:
            max_allowed_risk = (sizing_value * plan.leverage) * max_risk_pct

        step_size = exchange_filters.get("step_size", 0.0)
        min_notional_filter = exchange_filters.get("min_notional", 0.0)

        step_rounding_deviation = step_size * current_price
        notional_tolerance = max(step_rounding_deviation, 0.01)

        stop_distance_bps = (
            abs(current_price - proxy_stop) / current_price * 10000
            if proxy_stop > 0
            else 0
        )
        rounding_on_loss = step_size * current_price * stop_distance_bps / 10000
        risk_tolerance = (
            expected_entry_fee + expected_exit_fee + expected_slippage + rounding_on_loss
        )

        return ExecutableTrade(
            symbol=context.symbol,
            side=context.side,
            trade_side=trade_side,
            execution_id=context.execution_id,
            trade_group_id=context.trade_group_id,
            opportunity_id=context.opportunity_id,
            plan=plan,
            quantity=qty,
            quantity_str=qty_str,
            entry_price=current_price,
            requested_stake=sizing_value,
            leverage=plan.leverage,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            stop_price=proxy_stop,
            tp_price=proxy_tp,
            expected_notional=expected_notional,
            expected_loss=expected_loss,
            expected_reward=expected_reward,
            expected_entry_fee=expected_entry_fee,
            expected_exit_fee=expected_exit_fee,
            worst_case_loss=worst_case_loss,
            max_allowed_risk=max_allowed_risk,
            step_size=step_size,
            tick_size=exchange_filters.get("tick_size", 0.0),
            min_qty=exchange_filters.get("min_qty", 0.0),
            max_qty=exchange_filters.get("max_qty", 0.0),
            min_notional=min_notional_filter,
            notional_tolerance=notional_tolerance,
            risk_tolerance=risk_tolerance,
            available_balance=available_balance,
            atr=safe_float(indicators.get("atr"), 0.0),
        )

    def _emit_observation(
        self, category: str, importance: float, symbol: str, data: dict,
        position_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "source": "execution",
            "category": category,
            "importance": importance,
            "symbol": symbol,
            "data": data,
        }
        if position_id:
            payload["context"] = {"position_id": position_id}
        self._event_bus.publish_nowait(SystemEvent(
            event_type="OBSERVATION_EMITTED",
            service_name="ExecutionService",
            payload=payload,
        ))

    async def execute_entry(self, context: ExecutionContext) -> None:
        t_stage = perf_counter()

        quote_filter = self._config.get("universe", {}).get("quote_filter", "USDT")
        if quote_filter != "all":
            symbol_quote = "USDC" if context.symbol.endswith("USDC") else "USDT"
            if symbol_quote != quote_filter:
                logger.info(
                    "EXECUTION_SKIPPED_QUOTE_FILTER",
                    symbol=context.symbol,
                    symbol_quote=symbol_quote,
                    quote_filter=quote_filter,
                    trade_group_id=context.trade_group_id,
                )
                return

        logger.info(
            "EXECUTE_ENTRY_STARTED",
            symbol=context.symbol,
            side=context.side,
            execution_mode=context.execution_mode,
            trade_group_id=context.trade_group_id,
            opportunity_id=context.opportunity_id,
            elapsed_ms=0,
        )
        try:
            exec_cfg = self._config.get("execution", {})
            state = await self._context.get_state(context.symbol)
            current_price = state.get("current_price", 0.0)
            available_balance = 0.0
            qty_str, qty = await self._compute_sizing(
                context, current_price=current_price, exec_cfg=exec_cfg,
                available_balance=available_balance,
            )
            logger.info(
                "Sizing result",
                symbol=context.symbol,
                qty_str=qty_str,
                qty=qty,
                trade_group_id=context.trade_group_id,
                opportunity_id=context.opportunity_id,
                elapsed_ms=round((perf_counter() - t_stage) * 1000, 1),
            )
            if qty <= 0:
                logger.info(
                    "EXECUTION_PRECHECK_FAILED",
                    reason="quantity_zero",
                    symbol=context.symbol,
                    trade_group_id=context.trade_group_id,
                )
                return

            side = context.side
            trade_side = "LONG" if side == "BUY" else "SHORT"

            if context.execution_mode == "LIVE":
                async with self._live.client.symbol_execution_lock(context.symbol):
                    # ── Pre-trade exchange validations ──
                    exec_cfg = self._config.get("execution", {})

                    # Available balance check including fees and slippage buffer
                    leverage = int(exec_cfg.get("leverage", 10))
                    taker_fee_rate = await self._live.get_symbol_taker_fee(context.symbol)
                    notional_value = qty
                    initial_margin = notional_value / leverage
                    open_fee = notional_value * taker_fee_rate
                    slippage_buffer = 1.01
                    required_margin = (initial_margin + open_fee) * slippage_buffer
                    margin_asset = "USDC" if context.symbol.endswith("USDC") else "USDT"
                    available_balance = await self._live.get_available_balance(asset=margin_asset)

                    # Re-compute sizing with actual available balance for risk_pct mode
                    if exec_cfg.get("sizing_mode", "fixed_usdt") == "risk_pct" and available_balance > 0:
                        qty_str, qty = await self._compute_sizing(
                            context, current_price=current_price, exec_cfg=exec_cfg,
                            available_balance=available_balance,
                        )
                        notional_value = qty
                        initial_margin = notional_value / leverage
                        open_fee = notional_value * taker_fee_rate
                        required_margin = (initial_margin + open_fee) * slippage_buffer

                    # Diagnostic: full account snapshot
                    try:
                        acct = await self._live._client.get_account_info()
                        logger.info(
                            "ACCOUNT_BALANCE_SNAPSHOT",
                            symbol=context.symbol,
                            all_assets=[
                                {a["asset"]: {
                                    "wallet": a.get("walletBalance"),
                                    "available": a.get("availableBalance"),
                                    "cross_unrealized": a.get("crossUnPnl"),
                                    "cross_margin": a.get("crossWallet"),
                                }}
                                for a in acct.get("assets", [])
                                if float(a.get("walletBalance", 0)) > 0 or float(a.get("availableBalance", 0)) > 0
                            ],
                            open_positions=[
                                {p.get("symbol"): {
                                    "amt": p.get("positionAmt"),
                                    "entry": p.get("entryPrice"),
                                    "pnl": p.get("unRealizedProfit"),
                                    "leverage": p.get("leverage"),
                                }}
                                for p in acct.get("positions", [])
                                if float(p.get("positionAmt", 0)) != 0
                            ],
                        )
                    except Exception as acct_err:
                        logger.warning("Failed to fetch account snapshot", error=str(acct_err), exc_info=True)

                    logger.info(
                        "Margin check",
                        symbol=context.symbol,
                        required=round(required_margin, 4),
                        available=round(available_balance, 4),
                        margin_asset=margin_asset,
                        notional=round(notional_value, 4),
                        leverage=leverage,
                        trade_group_id=context.trade_group_id,
                        elapsed_ms=round((perf_counter() - t_stage) * 1000, 1),
                    )
                    if required_margin > available_balance:
                        raise InsufficientMarginError(
                            f"Cannot execute entry. Required: {required_margin:.4f} {margin_asset}, "
                            f"Available: {available_balance:.4f} {margin_asset} (Includes fee buffer)"
                        )

                    # ── Build ExecutionPlan and ExecutableTrade ──
                    hard_stop_side = "SELL" if trade_side == "LONG" else "BUY"
                    stop_loss_pct_from_cfg = float(exec_cfg.get("stop_loss_pct", 0.98))
                    take_profit_pct_from_cfg = float(exec_cfg.get("take_profit_pct", 1.04))
                    proxy_stop = current_price * stop_loss_pct_from_cfg
                    proxy_tp = current_price * take_profit_pct_from_cfg
                    if trade_side == "SHORT":
                        proxy_stop = current_price * (2.0 - stop_loss_pct_from_cfg)
                        proxy_tp = current_price * (2.0 - take_profit_pct_from_cfg)
                    exchange_filters = await self._live._client.get_symbol_filters(context.symbol)
                    state = await self._context.get_state(context.symbol)
                    indicators = state.get("indicators", {})

                    plan = self._build_execution_plan(context, exec_cfg)
                    trade = self._build_executable_trade(
                        context=context, qty=qty, qty_str=qty_str,
                        current_price=current_price, trade_side=trade_side,
                        proxy_stop=proxy_stop, proxy_tp=proxy_tp,
                        exec_cfg=exec_cfg, exchange_filters=exchange_filters,
                        taker_fee_rate=taker_fee_rate,
                        available_balance=available_balance,
                        indicators=indicators, plan=plan,
                    )

                    # ── Three-validator pipeline ──
                    consistency_outcomes = ConsistencyValidator.validate(trade)
                    intent_outcomes = IntentValidator.validate(trade)
                    exchange_outcomes = ExchangeValidator.validate(trade)

                    report = TradeValidationReport(
                        phase="pre_submit",
                        consistency=consistency_outcomes,
                        intent=intent_outcomes,
                        exchange=exchange_outcomes,
                    )

                    if report.passed:
                        self._publish_audit_event(
                            "TRADE_VALIDATED",
                            context.symbol, context.execution_id,
                            {"expected_notional": trade.expected_notional,
                             "expected_loss": trade.expected_loss,
                             "worst_case_loss": trade.worst_case_loss,
                             "notional_tolerance": trade.notional_tolerance,
                             "risk_tolerance": trade.risk_tolerance},
                        )

                    all_outcomes = consistency_outcomes + intent_outcomes + exchange_outcomes
                    for outcome in all_outcomes:
                        self._publish_audit_event(
                            "SAFETY_VALIDATION_FAILED",
                            context.symbol, context.execution_id,
                            {"code": outcome.code, "message": outcome.message,
                             "status": outcome.status.name},
                        )
                        if outcome.status == ValidationOutcomeStatus.FATAL_FAILURE:
                            logger.critical(
                                "VALIDATION_REJECTED",
                                symbol=context.symbol,
                                code=outcome.code,
                                message=outcome.message,
                            )

                    if not report.passed:
                        self._publish_audit_event(
                            "TRADE_REJECTED",
                            context.symbol, context.execution_id,
                            {"failures": [o.code for o in all_outcomes
                                          if o.status == ValidationOutcomeStatus.FATAL_FAILURE]},
                        )
                        logger.info(
                            "EXECUTION_VALIDATION_FAILED",
                            reason="validation_rejected",
                            symbol=context.symbol,
                            failures=[o.code for o in all_outcomes
                                      if o.status == ValidationOutcomeStatus.FATAL_FAILURE],
                        )
                        return

                    config_usdt = float(exec_cfg.get("sizing_value", 2.0)) * int(exec_cfg.get("leverage", 10))
                    tolerance = max(1.0, config_usdt * 0.05)
                    if abs(qty - config_usdt) > tolerance:
                        logger.critical(
                            "SIZING_MISMATCH_ABORT",
                            symbol=context.symbol,
                            computed_notional=qty,
                            config_notional=config_usdt,
                            tolerance=tolerance,
                            trade_group_id=context.trade_group_id,
                            opportunity_id=context.opportunity_id,
                        )
                        self._publish_audit_event(
                            "SIZING_MISMATCH_ABORT",
                            context.symbol, context.execution_id,
                            {"computed_notional": qty, "config_notional": config_usdt},
                        )
                        logger.info(
                            "EXECUTION_PRECHECK_FAILED",
                            reason="sizing_mismatch",
                            symbol=context.symbol,
                            trade_group_id=context.trade_group_id,
                        )
                        return

                    # ── Write-ahead: persist EXECUTING position before market order ──
                    pending_pos = Position(
                        position_id=context.execution_id,
                        symbol=context.symbol,
                        side=trade_side,
                        quantity=qty,
                        avg_fill_price=0.0,
                        anchor_symbol=context.anchor_symbol,
                        lifecycle_state=PositionState.EXECUTING,
                        execution_mode=context.execution_mode,
                        execution_id=context.execution_id,
                        trade_group_id=context.trade_group_id,
                        candidate_id=context.candidate_id,
                        correlation_id=context.correlation_id,
                        llm_request_id=context.llm_request_id,
                        strategy_version=context.strategy_version,
                        execution_model=context.execution_model,
                        execution_model_version=context.execution_model_version,
                        execution_parameters=dict(context.execution_parameters),
                        risk_decision=context.risk_decision,
                        risk_decision_reason=context.risk_decision_reason,
                        entry_thesis=context.entry_thesis,
                        timeframe=context.timeframe,
                        max_holding_period_minutes=context.max_holding_period_minutes,
                        opportunity_id=context.opportunity_id,
                        active_profile_id=context.active_profile_id,
                        session_id=context.session_id,
                        entry_timestamp=context.entry_timestamp,
                    )
                    await self._portfolio.add_position(pending_pos)
                    logger.info(
                        "EXECUTING_POSITION_PERSISTED",
                        position_id=context.execution_id,
                        symbol=context.symbol,
                        side=trade_side,
                        qty=qty,
                    )

                    logger.info(
                        "API_ORDER_REQUEST",
                        symbol=context.symbol,
                        side=side,
                        qty=qty_str,
                        trade_group_id=context.trade_group_id,
                        opportunity_id=context.opportunity_id,
                        execution_mode=context.execution_mode,
                        elapsed_ms=round((perf_counter() - t_stage) * 1000, 1),
                    )
                    auth = await self._live.execute_entry(
                        context.symbol, side, qty_str, context,
                        current_price=current_price,
                    )
                    logger.info(
                        "API_ORDER_RESPONSE",
                        symbol=context.symbol,
                        order_id=auth.get("orderId"),
                        status=auth.get("status"),
                        executed_qty=auth.get("executedQty"),
                        avg_price=auth.get("avgPrice"),
                        trade_group_id=context.trade_group_id,
                        elapsed_ms=round((perf_counter() - t_stage) * 1000, 1),
                    )
            else:
                # ── Write-ahead: persist EXECUTING position before virtual entry ──
                pending_pos = Position(
                    position_id=context.execution_id,
                    symbol=context.symbol,
                    side=trade_side,
                    quantity=qty,
                    avg_fill_price=0.0,
                    anchor_symbol=context.anchor_symbol,
                    lifecycle_state=PositionState.EXECUTING,
                    execution_mode=context.execution_mode,
                    execution_id=context.execution_id,
                    trade_group_id=context.trade_group_id,
                    candidate_id=context.candidate_id,
                    correlation_id=context.correlation_id,
                    llm_request_id=context.llm_request_id,
                    strategy_version=context.strategy_version,
                    execution_model=context.execution_model,
                    execution_model_version=context.execution_model_version,
                    execution_parameters=dict(context.execution_parameters),
                    risk_decision=context.risk_decision,
                    risk_decision_reason=context.risk_decision_reason,
                    entry_thesis=context.entry_thesis,
                    timeframe=context.timeframe,
                    max_holding_period_minutes=context.max_holding_period_minutes,
                    opportunity_id=context.opportunity_id,
                    active_profile_id=context.active_profile_id,
                    session_id=context.session_id,
                    entry_timestamp=context.entry_timestamp,
                )
                await self._portfolio.add_position(pending_pos)
                logger.info(
                    "EXECUTING_POSITION_PERSISTED",
                    position_id=context.execution_id,
                    symbol=context.symbol,
                    side=trade_side,
                    qty=qty,
                )
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
                    trade_group_id=context.trade_group_id,
                )
                logger.info(
                    "EXECUTION_API_REJECTED",
                    reason="invalid_fill",
                    symbol=context.symbol,
                    trade_group_id=context.trade_group_id,
                )
                return

            # ── Post-fill drift detection ──
            if context.execution_mode == "LIVE":
                expected_notional = float(qty_str)
                actual_notional = executed_qty * avg_price
                drift_pct = 0.0
                if expected_notional > 0:
                    drift_pct = abs(actual_notional - expected_notional) / expected_notional * 100
                DRIFT_TOLERANCE_PCT = 5.0
                if drift_pct > DRIFT_TOLERANCE_PCT:
                    logger.warning(
                        "EXECUTION_DRIFT_DETECTED",
                        symbol=context.symbol,
                        expected_notional=round(expected_notional, 4),
                        actual_notional=round(actual_notional, 4),
                        drift_pct=round(drift_pct, 2),
                        avg_fill_price=avg_price,
                    )
                    self._publish_audit_event(
                        "EXECUTION_DRIFT_DETECTED",
                        context.symbol, context.execution_id,
                        {
                            "expected_notional": expected_notional,
                            "actual_notional": actual_notional,
                            "drift_pct": round(drift_pct, 2),
                            "avg_fill_price": avg_price,
                        },
                    )

            exec_cfg = self._config.get("execution", {})
            stop_loss_pct = float(exec_cfg.get("stop_loss_pct", 0.98))
            take_profit_pct = float(exec_cfg.get("take_profit_pct", 1.04))

            if trade_side == "LONG":
                stop_loss = avg_price * stop_loss_pct
                take_profit = avg_price * take_profit_pct
            else:
                stop_loss = avg_price * (2.0 - stop_loss_pct)
                take_profit = avg_price * (2.0 - take_profit_pct)

            sizing_value = float(self._config.get("execution", {}).get("sizing_value", 5.0))
            effective_leverage = int(self._config.get("execution", {}).get("leverage", 10))
            effective_max_risk_pct = float(self._config.get("execution", {}).get("max_risk_pct", 0.02))
            implied_loss = abs(avg_price - stop_loss) * executed_qty
            sizing_mode = self._config.get("execution", {}).get("sizing_mode", "fixed_usdt")
            if sizing_mode == "risk_pct":
                expected_max_loss = available_balance * effective_max_risk_pct
            else:
                expected_max_loss = (sizing_value * effective_leverage) * effective_max_risk_pct
            if implied_loss > expected_max_loss and expected_max_loss > 0:
                logger.error(
                    "IMPLIED_LOSS_EXCEEDS_RISK_BUDGET",
                    symbol=context.symbol,
                    implied_loss=round(implied_loss, 4),
                    expected_max_loss=round(expected_max_loss, 4),
                    sizing_value=sizing_value,
                )
                self._publish_audit_event(
                    "IMPLIED_LOSS_EXCEEDS_RISK_BUDGET",
                    context.symbol, context.execution_id,
                    {"implied_loss": implied_loss, "expected_max_loss": expected_max_loss},
                )

            if context.execution_mode == "LIVE":
                actual_notional = executed_qty * avg_price
                sizing_mode = self._config.get("execution", {}).get("sizing_mode", "fixed_usdt")
                if sizing_mode == "fixed_usdt":
                    expected_position = sizing_value * leverage
                    tolerance = max(1.0, expected_position * 0.2)
                    if abs(actual_notional - expected_position) > tolerance:
                        logger.critical(
                            "SIZING_MISMATCH_FORCE_FLAT",
                            symbol=context.symbol,
                            expected_notional=expected_position,
                            actual_notional=round(actual_notional, 2),
                            tolerance=tolerance,
                        )
                        self._publish_audit_event(
                            "SIZING_MISMATCH_FORCE_FLAT",
                            context.symbol, context.execution_id,
                            {"expected_notional": expected_position, "actual_notional": actual_notional},
                        )
                        close_side = "SELL" if trade_side == "LONG" else "BUY"
                        close_qty_str = await self._live.round_quantity(context.symbol, executed_qty, round_up=False)
                        try:
                            flat_result = await self._live._client.place_market_order(
                                symbol=context.symbol,
                                side=close_side,
                                quantity=close_qty_str,
                                position_side="BOTH",
                            )
                            logger.critical(
                                "FORCE_FLAT_SUBMITTED",
                                symbol=context.symbol,
                                flat_qty=close_qty_str,
                                flat_order_id=flat_result.get("orderId"),
                            )
                        except Exception as flat_err:
                            logger.critical(
                                "FORCE_FLAT_FAILED",
                                symbol=context.symbol,
                                error=str(flat_err),
                            )
                        return

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
                        "Recording position locally with FAILED protection — repair on next cycle.",
                        symbol=context.symbol, error=str(e),
                    )
                    sid = self._live._short_id(context.execution_id)
                    protection_data = {
                        "stop_order_id": None,
                        "stop_client_order_id": f"SL_{sid}",
                        "tp_order_id": None,
                        "tp_client_order_id": f"TP_{sid}",
                        "stop_price": stop_loss,
                        "tp_price": take_profit,
                        "status": "FAILED",
                    }
                    event = SystemEvent(
                        event_type="PROTECTION_FAILED",
                        service_name="ExecutionService",
                        payload={
                            "symbol": context.symbol,
                            "execution_id": context.execution_id,
                            "error": str(e),
                        },
                    )
                    await self._event_bus.publish(event)
                    self._emit_observation(
                        "risk", 0.75, context.symbol,
                        {"event": "protection_failed", "error": str(e)},
                    )
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
                "position_id": context.execution_id,
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
                "timeframe": context.timeframe,
                "max_holding_period_minutes": getattr(context, "max_holding_period_minutes", 0.0),
                "opportunity_id": context.opportunity_id,
                "active_profile_id": getattr(context, "active_profile_id", None),
                "session_id": getattr(context, "session_id", None),
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
            self._emit_observation(
                "execution", 0.70, context.symbol,
                {"event": "entry_executed", "mode": context.execution_mode,
                 "side": context.side, "qty": executed_qty, "price": avg_price,
                 "timeframe": context.timeframe},
                context.execution_id,
            )
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
            logger.info(
                "EXECUTION_SUCCESS",
                symbol=context.symbol,
                side=trade_side,
                qty=executed_qty,
                price=avg_price,
                trade_group_id=context.trade_group_id,
                elapsed_ms=round((perf_counter() - t_stage) * 1000, 1),
            )

            if context.execution_mode == "LIVE" and self._mirror_enabled:
                await self._create_mirror(context, avg_price, stop_loss, take_profit, executed_qty)

        except InsufficientMarginError as e:
            logger.warning(
                "Insufficient margin — entry skipped",
                symbol=context.symbol,
                execution_mode=context.execution_mode,
                error=str(e), exc_info=True,
            )
            logger.info(
                "EXECUTION_PRECHECK_FAILED",
                reason="insufficient_margin",
                symbol=context.symbol,
                trade_group_id=context.trade_group_id,
            )
        except BreakerTripped as e:
            logger.warning(
                "CIRCUIT_BREAKER_TRIPPED — order rejected by exposure breaker",
                symbol=context.symbol,
                execution_mode=context.execution_mode,
                error=str(e),
            )
            self._publish_audit_event(
                "CIRCUIT_BREAKER_TRIPPED",
                context.symbol, context.execution_id,
                {"error": str(e)},
            )
            logger.info(
                "EXECUTION_API_REJECTED",
                reason="circuit_breaker_tripped",
                symbol=context.symbol,
                trade_group_id=context.trade_group_id,
            )
        except Exception as e:
            logger.error(
                "Entry execution failed",
                symbol=context.symbol,
                execution_mode=context.execution_mode,
                error=str(e), exc_info=True,
            )
            logger.info(
                "EXECUTION_API_REJECTED",
                reason="exception",
                symbol=context.symbol,
                trade_group_id=context.trade_group_id,
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
                "timeframe": source_context.timeframe,
                "max_holding_period_minutes": getattr(source_context, "max_holding_period_minutes", 0.0),
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
                "opportunity_id": source_context.opportunity_id,
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
            self._emit_observation(
                "execution", 0.70, source_context.symbol,
                {"event": "entry_executed", "mode": "SHADOW",
                 "side": source_context.side, "qty": qty, "price": mirror_price,
                 "timeframe": source_context.timeframe},
                mirror_context.execution_id,
            )
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
            self._emit_observation(
                "execution", 0.80, position.symbol,
                {"event": "exit_executed", "reason": reason, "exit_price": avg_price,
                 "execution_mode": position.execution_mode},
                position.position_id,
            )
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
            failure_statuses = {"transient_failure", "safety_failed", "price_invalid"}
            is_failure = not result or (isinstance(result, dict) and result.get("status") in failure_statuses)
            if is_failure:
                pos_id = position.position_id
                trail_count = self._trailing_failures.get(pos_id, 0) + 1
                reason = result.get("status") if result else "empty"
                self._publish_audit_event(
                    "TRAILING_FAILED", position.symbol, pos_id,
                    {"reason": reason, "new_stop_price": new_stop_price, "retry": trail_count},
                )
                logger.warning(
                    "Trailing stop update aborted or failed, not updating local position state",
                    position_id=pos_id,
                    symbol=position.symbol,
                    reason=reason,
                    retry=trail_count,
                )
                should_close = (
                    isinstance(result, dict) and result.get("status") == "safety_failed"
                ) or trail_count >= 3
                if should_close:
                    self._trailing_failures.pop(pos_id, None)
                    logger.critical(
                        "TRAILING_FAILED_EXHAUSTED",
                        position_id=pos_id,
                        symbol=position.symbol,
                        reason=reason,
                        retry=trail_count,
                    )
                    try:
                        exchange_positions = await self._live._client.get_open_positions()
                        ex_pos = next(
                            (p for p in exchange_positions if p.get("symbol") == position.symbol),
                            None,
                        )
                        if ex_pos is None:
                            logger.critical(
                                "TRAILING_FORCE_CLOSE_SKIPPED: no exchange position",
                                position_id=pos_id,
                                symbol=position.symbol,
                            )
                        else:
                            amt = float(ex_pos.get("positionAmt", 0))
                            if amt == 0:
                                logger.critical(
                                    "TRAILING_FORCE_CLOSE_SKIPPED: exchange position already flat",
                                    position_id=pos_id,
                                    symbol=position.symbol,
                                )
                            else:
                                await self._live._client.force_close_position(position.symbol, amt)
                    except Exception as fe:
                        logger.critical(
                            "TRAILING_FORCE_CLOSE_FAILED",
                            position_id=pos_id,
                            symbol=position.symbol,
                            error=str(fe),
                        )
                else:
                    self._trailing_failures[pos_id] = trail_count
                self._emit_observation(
                    "position", 0.45, position.symbol,
                    {"event": "trailing_stop_failed", "reason": reason, "retry": trail_count},
                    position.position_id,
                )
                return {}
            self._trailing_failures.pop(position.position_id, None)
            self._emit_observation(
                "position", 0.50, position.symbol,
                {"event": "trailing_stop_replaced", "new_stop": new_stop_price},
                position.position_id,
            )
            self._publish_audit_event(
                "TRAILING_REPLACED", position.symbol, position.position_id,
                {"new_stop_price": new_stop_price,
                 "order_id": result.get("order_id"),
                 "client_order_id": result.get("client_order_id")},
            )
            # Preserve TP fields — never clear them during trailing
            existing_tp_price = position.protection_orders.tp_price
            existing_tp_order_id = position.protection_orders.tp_order_id
            existing_tp_client_order_id = position.protection_orders.tp_client_order_id

            position.protection_orders.stop_price = new_stop_price
            position.protection_orders.stop_order_id = str(result.get("order_id", ""))
            position.protection_orders.stop_client_order_id = result.get("client_order_id", "")
            position.protection_orders.last_updated = datetime.utcnow()
            position.protection_orders.status = "UPDATED"

            position.protection_orders.tp_price = existing_tp_price
            position.protection_orders.tp_order_id = existing_tp_order_id
            position.protection_orders.tp_client_order_id = existing_tp_client_order_id
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

    async def get_open_protection_ids(self, symbol: str) -> set[str]:
        if self._live is None:
            return set()
        try:
            orders = await self._live._client.get_open_algo_orders(symbol)
            ids: set[str] = set()
            for o in orders:
                client_id = o.get("clientAlgoId")
                algo_id = o.get("algoId")
                if client_id:
                    ids.add(str(client_id))
                if algo_id is not None:
                    ids.add(str(algo_id))
            return ids
        except Exception:
            return set()

    def _publish_audit_event(
        self, event_type: str, symbol: str, position_id: str, extra: dict = None,
    ) -> None:
        payload = {
            "symbol": symbol,
            "position_id": position_id,
            **(extra or {}),
        }
        event = SystemEvent(
            event_type=event_type,
            service_name="ExecutionService",
            payload=payload,
        )
        self._event_bus.publish_nowait(event)
        self._portfolio._store.append_audit_log(event)
