from __future__ import annotations

import asyncio
import math
import uuid

import structlog

from src.api.binance_client import BinanceClient
from src.core.models import Position, PositionState, SystemEvent
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("reconciler")

DUCKDB_QTY_TOLERANCE = 1e-8


def _short_id(position_id: str) -> str:
    return position_id.replace("-", "")[:16]


def _algo_order_type(order: dict) -> str:
    return order.get("type") or order.get("orderType") or ""


def _close_side_for_position(side: str) -> str:
    return "SELL" if side == "LONG" else "BUY"


def _find_stop_order(algo_orders: list[dict], close_side: str) -> dict | None:
    return next(
        (o for o in algo_orders if _algo_order_type(o) == "STOP_MARKET" and o.get("side") == close_side),
        None,
    )


def _find_tp_order(algo_orders: list[dict], close_side: str) -> dict | None:
    return next(
        (
            o for o in algo_orders
            if _algo_order_type(o) == "TAKE_PROFIT_MARKET" and o.get("side") == close_side
        ),
        None,
    )


def _protection_id(value) -> str | None:
    if value is None:
        return None
    return str(value)


async def _place_or_link_stop(
    client: BinanceClient,
    symbol: str,
    close_side: str,
    stop_price: float,
    position_id: str,
    qty: float,
    mark_price: float,
    stop_cid: str | None = None,
) -> tuple[float, str | None, str | None]:
    """Place a stop or link an existing one. Returns (price, algo_id, client_id)."""
    try:
        algo_orders = await client.get_open_algo_orders(symbol)
    except Exception:
        algo_orders = []

    existing = _find_stop_order(algo_orders, close_side)
    if existing is not None:
        return (
            float(existing.get("triggerPrice", stop_price)),
            _protection_id(existing.get("algoId")),
            existing.get("clientAlgoId") or stop_cid,
        )

    try:
        stop_resp = await client.place_algo_stop(
            symbol, close_side, stop_price, position_id,
            client_algo_id=stop_cid,
            estimated_qty=qty, current_price=mark_price,
        )
        return (
            stop_price,
            _protection_id(stop_resp.get("algoId")),
            stop_cid or f"SL_{_short_id(position_id)}",
        )
    except Exception as e:
        if "-4130" not in str(e):
            raise
        algo_orders = await client.get_open_algo_orders(symbol)
        existing = _find_stop_order(algo_orders, close_side)
        if existing is None:
            raise
        logger.info(
            "-4130: Linked to existing stop on exchange",
            symbol=symbol,
            algo_id=existing.get("algoId"),
            client_algo_id=existing.get("clientAlgoId"),
        )
        return (
            float(existing.get("triggerPrice", stop_price)),
            _protection_id(existing.get("algoId")),
            existing.get("clientAlgoId") or stop_cid,
        )


async def _place_or_link_tp(
    client: BinanceClient,
    symbol: str,
    close_side: str,
    tp_price: float,
    position_id: str,
    qty: float,
    mark_price: float,
    tp_cid: str | None = None,
) -> tuple[float, str | None, str | None]:
    """Place a TP or link an existing one. Returns (price, algo_id, client_id)."""
    try:
        algo_orders = await client.get_open_algo_orders(symbol)
    except Exception:
        algo_orders = []

    existing = _find_tp_order(algo_orders, close_side)
    if existing is not None:
        return (
            float(existing.get("triggerPrice", tp_price)),
            _protection_id(existing.get("algoId")),
            existing.get("clientAlgoId") or tp_cid,
        )

    try:
        tp_resp = await client.place_algo_tp(
            symbol, close_side, tp_price, position_id,
            client_algo_id=tp_cid,
            estimated_qty=qty, current_price=mark_price,
        )
        return (
            tp_price,
            _protection_id(tp_resp.get("algoId")),
            tp_cid or f"TP_{_short_id(position_id)}",
        )
    except Exception as e:
        if "-4130" not in str(e):
            raise
        algo_orders = await client.get_open_algo_orders(symbol)
        existing = _find_tp_order(algo_orders, close_side)
        if existing is None:
            raise
        logger.info(
            "-4130: Linked to existing TP on exchange",
            symbol=symbol,
            algo_id=existing.get("algoId"),
            client_algo_id=existing.get("clientAlgoId"),
        )
        return (
            float(existing.get("triggerPrice", tp_price)),
            _protection_id(existing.get("algoId")),
            existing.get("clientAlgoId") or tp_cid,
        )


class Reconciler:
    @staticmethod
    async def sync_missing_positions_from_exchange(
        client: BinanceClient,
        portfolio_mgr: PortfolioManager,
        take_profit_pct: float = 1.04,
    ) -> None:
        """Mid-run adoption: sync any exchange positions missing from local DB."""
        exchange_positions = await client.get_open_positions()

        for ex_pos in exchange_positions:
            symbol = ex_pos["symbol"]

            exchange_qty = ex_pos["position_amt"]
            async with client.symbol_execution_lock(symbol):
                # Re-query local state inside the lock to avoid adopting
                # a position that was just created by a concurrent entry.
                if any(p.symbol == symbol for p in portfolio_mgr.get_open_positions()):
                    continue

                mark_price = ex_pos.get("mark_price", ex_pos["entry_price"])
                side = "LONG" if exchange_qty > 0 else "SHORT"
                new_pos_id = str(uuid.uuid4())

                logger.info(
                    "Adopting missing exchange position into local DB",
                    symbol=symbol, side=side, quantity=abs(exchange_qty),
                    entry_price=ex_pos["entry_price"],
                )

                # Check for existing algo protection on exchange
                close_side = _close_side_for_position(side)
                existing_stop = None
                try:
                    algo_orders = await client.get_open_algo_orders(symbol)
                    existing_stop = _find_stop_order(algo_orders, close_side)
                except Exception:
                    logger.warning("Failed to query existing algo orders", symbol=symbol, exc_info=True)

                if existing_stop is not None:
                    existing_price = float(existing_stop.get("triggerPrice", 0.0))
                    existing_cid = existing_stop.get("clientAlgoId", "")
                    existing_aid = _protection_id(existing_stop.get("algoId"))

                    new_pos = Position(
                        position_id=new_pos_id,
                        symbol=symbol,
                        side=side,
                        quantity=abs(exchange_qty),
                        avg_fill_price=mark_price,
                        entry_thesis="UNMANAGED_POSITION_ADOPTED_FOR_RISK_PROTECTION",
                        anchor_symbol=symbol,
                        initial_stop_loss=existing_price,
                        current_stop=existing_price,
                        initial_take_profit=0.0,
                        current_target=0.0,
                        lifecycle_state=PositionState.UNMANAGED_ADOPTED,
                    )
                    new_pos.protection_orders.stop_price = existing_price
                    new_pos.protection_orders.stop_order_id = existing_aid
                    new_pos.protection_orders.stop_client_order_id = existing_cid

                    # Link or place TP
                    existing_tp = None
                    try:
                        algo_orders = await client.get_open_algo_orders(symbol)
                        existing_tp = _find_tp_order(algo_orders, close_side)
                    except Exception:
                        logger.warning("Failed to query existing TP orders", symbol=symbol, exc_info=True)

                    if existing_tp is not None:
                        new_pos.protection_orders.tp_price = float(existing_tp.get("triggerPrice", 0.0))
                        new_pos.protection_orders.tp_order_id = _protection_id(existing_tp.get("algoId"))
                        new_pos.protection_orders.tp_client_order_id = existing_tp.get("clientAlgoId")
                    else:
                        emergency_tp = mark_price * take_profit_pct if side == "LONG" else mark_price * (2.0 - take_profit_pct)
                        try:
                            tp_price, tp_aid, tp_cid = await _place_or_link_tp(
                                client, symbol, close_side, emergency_tp, new_pos_id,
                                abs(exchange_qty), mark_price,
                            )
                            new_pos.protection_orders.tp_price = tp_price
                            new_pos.protection_orders.tp_order_id = tp_aid
                            new_pos.protection_orders.tp_client_order_id = tp_cid
                        except Exception as e:
                            logger.error("Failed to place TP for adopted position", symbol=symbol, error=str(e))

                    await portfolio_mgr.add_position(new_pos)
                    logger.info(
                        "Adopted position and linked to existing exchange protection",
                        symbol=symbol, position_id=new_pos_id,
                        client_algo_id=existing_cid, stop_price=existing_price,
                    )
                else:
                    # Use config-based prices — don't place emergency orders directly.
                    # The protection audit/repair cycle will place them.
                    sl_price = mark_price * 0.98 if side == "LONG" else mark_price * (2.0 - 0.98)
                    tp_price = mark_price * take_profit_pct if side == "LONG" else mark_price * (2.0 - take_profit_pct)

                    new_pos = Position(
                        position_id=new_pos_id,
                        symbol=symbol,
                        side=side,
                        quantity=abs(exchange_qty),
                        avg_fill_price=mark_price,
                        entry_thesis="UNMANAGED_POSITION_ADOPTED_FOR_RISK_PROTECTION",
                        anchor_symbol=symbol,
                        initial_stop_loss=sl_price,
                        current_stop=sl_price,
                        initial_take_profit=tp_price,
                        current_target=tp_price,
                        lifecycle_state=PositionState.UNMANAGED_ADOPTED,
                    )
                    new_pos.protection_orders.stop_price = sl_price
                    new_pos.protection_orders.stop_client_order_id = f"SL_{_short_id(new_pos_id)}"
                    new_pos.protection_orders.tp_price = tp_price
                    new_pos.protection_orders.tp_client_order_id = f"TP_{_short_id(new_pos_id)}"
                    new_pos.protection_orders.status = "PENDING"

                    await portfolio_mgr.add_position(new_pos)
                    logger.info(
                        "Adopted position — protection pending audit/repair cycle",
                        symbol=symbol, position_id=new_pos_id,
                        stop_price=sl_price, tp_price=tp_price,
                    )

    @staticmethod
    async def reconcile(
        portfolio_mgr: PortfolioManager,
        client: BinanceClient,
        max_positions: int = 3,
        take_profit_pct: float = 1.04,
    ) -> dict:
        logger.info("Starting exchange-to-local position reconciliation")

        # --- Step 1: Query exchange for open positions ---
        exchange_positions = await client.get_open_positions()
        exchange_by_symbol: dict[str, dict] = {
            p["symbol"]: p for p in exchange_positions
        }

        # --- Step 2: Query exchange for open orders (regular + algo) ---
        exchange_orders = await client.get_open_orders()
        open_order_symbols = {o.get("symbol") for o in exchange_orders if o.get("symbol")}

        algo_orders = await client.get_open_algo_orders()
        algo_by_symbol: dict[str, list[dict]] = {}
        for o in algo_orders:
            sym = o.get("symbol")
            if sym:
                algo_by_symbol.setdefault(sym, []).append(o)

        # --- Step 3: Query local open LIVE positions (shadow has no exchange counterpart) ---
        local_positions = portfolio_mgr.get_live_positions()
        local_by_symbol: dict[str, Position] = {
            p.symbol: p for p in local_positions
        }

        results = {
            "orphaned_closed": 0,
            "adopted": 0,
            "qty_mismatches": 0,
            "exchange_positions": len(exchange_positions),
            "local_open_positions": len(local_positions),
            "open_orders": len(exchange_orders),
            "open_algo_orders": len(algo_orders),
            "open_order_symbols": sorted(open_order_symbols),
            "details": [],
        }

        # --- Step 4a: ORPHANED — DB has OPEN, exchange has NO position ---
        for symbol, local_pos in local_by_symbol.items():
            exchange_pos = exchange_by_symbol.get(symbol)
            if exchange_pos is None:
                logger.warning(
                    "Local position is OPEN but no exchange position exists. "
                    "Marking as ORPHANED_RECONCILIATION.",
                    symbol=symbol,
                    position_id=local_pos.position_id,
                    local_qty=local_pos.quantity,
                )
                # Persist exit data for learning pipeline. The position is saved with
                # CLOSED state + exit_reason; a startup sweep can build the full LearningManifest later.
                portfolio_mgr._store.append_audit_log(SystemEvent(
                    event_type="POSITION_CLOSED_ORPHANED",
                    service_name="Reconciler",
                    payload={
                        "position_id": local_pos.position_id,
                        "symbol": symbol,
                        "side": local_pos.side,
                        "quantity": local_pos.quantity,
                        "entry_price": local_pos.avg_fill_price,
                        "current_stop": local_pos.current_stop,
                        "exit_reason": "ORPHANED_RECONCILIATION",
                        "protection_status": local_pos.protection_orders.status,
                    },
                ))

                try:
                    await portfolio_mgr.update_position_state(
                        local_pos.position_id,
                        PositionState.CLOSING,
                    )
                    await portfolio_mgr.update_position_state(
                        local_pos.position_id,
                        PositionState.CLOSED,
                        exit_reason="ORPHANED_RECONCILIATION",
                    )
                    results["orphaned_closed"] += 1
                    logger.info(
                        "Orphaned position closed in DB",
                        symbol=symbol,
                        position_id=local_pos.position_id,
                    )
                    results["details"].append({
                        "type": "ORPHANED",
                        "symbol": symbol,
                        "position_id": local_pos.position_id,
                        "local_qty": local_pos.quantity,
                    })
                except Exception as e:
                    logger.error(
                        "Failed to close orphaned position",
                        symbol=symbol,
                        position_id=local_pos.position_id,
                        error=str(e),
                    )
                    results["details"].append({
                        "type": "ORPHANED_FAILED",
                        "symbol": symbol,
                        "position_id": local_pos.position_id,
                        "error": str(e),
                    })

        # --- Step 4b: UNMANAGED — exchange has position, DB has no OPEN ---
        local_count_before = len(local_by_symbol)
        adopted_count = 0
        for symbol, ex_pos in exchange_by_symbol.items():
            local_pos = local_by_symbol.get(symbol)
            if local_pos is None:
                # Enforce max_positions cap on adoption
                if (local_count_before + adopted_count) >= max_positions:
                    logger.warning(
                        "Max concurrent positions reached — skipping adoption",
                        symbol=symbol,
                        current_positions=local_count_before + adopted_count,
                        max_positions=max_positions,
                    )
                    results.setdefault("adoption_skipped", 0)
                    results["adoption_skipped"] += 1
                    results["details"].append({
                        "type": "ADOPTION_SKIPPED_LIMIT",
                        "symbol": symbol,
                        "exchange_qty": ex_pos["position_amt"],
                    })
                    continue

                exchange_qty = ex_pos["position_amt"]
                mark_price = ex_pos.get("mark_price", ex_pos["entry_price"])
                side = "LONG" if exchange_qty > 0 else "SHORT"
                logger.warning(
                    "Exchange position detected with no local record. "
                    "Adopting with config-based SL/TP — protection pending audit/repair.",
                    symbol=symbol,
                    exchange_qty=exchange_qty,
                    entry_price=ex_pos["entry_price"],
                    mark_price=mark_price,
                )

                try:
                    new_pos_id = str(uuid.uuid4())
                    close_side = _close_side_for_position(side)

                    # Scan exchange for existing algo orders before adopting
                    algo_ords = algo_by_symbol.get(symbol, [])
                    existing_stop = _find_stop_order(algo_ords, close_side)
                    existing_tp = _find_tp_order(algo_ords, close_side)

                    sl_price = mark_price * 0.98 if side == "LONG" else mark_price * (2.0 - 0.98)
                    tp_price = mark_price * take_profit_pct if side == "LONG" else mark_price * (2.0 - take_profit_pct)

                    new_pos = Position(
                        position_id=new_pos_id,
                        symbol=symbol,
                        side=side,
                        quantity=abs(exchange_qty),
                        avg_fill_price=mark_price,
                        entry_thesis="UNMANAGED_POSITION_ADOPTED_FOR_RISK_PROTECTION",
                        anchor_symbol=symbol,
                        initial_stop_loss=sl_price,
                        current_stop=sl_price,
                        initial_take_profit=tp_price,
                        current_target=tp_price,
                        lifecycle_state=PositionState.UNMANAGED_ADOPTED,
                    )

                    if existing_stop is not None:
                        new_pos.protection_orders.stop_price = float(existing_stop.get("triggerPrice", sl_price))
                        new_pos.protection_orders.stop_order_id = _protection_id(existing_stop.get("algoId"))
                        new_pos.protection_orders.stop_client_order_id = existing_stop.get("clientAlgoId") or f"SL_{_short_id(new_pos_id)}"
                        new_pos.protection_orders.status = "VERIFIED"
                        logger.info("Linked to existing exchange stop on adoption", symbol=symbol, algo_id=existing_stop.get("algoId"))
                    else:
                        new_pos.protection_orders.stop_price = sl_price
                        new_pos.protection_orders.stop_client_order_id = f"SL_{_short_id(new_pos_id)}"

                    if existing_tp is not None:
                        new_pos.protection_orders.tp_price = float(existing_tp.get("triggerPrice", tp_price))
                        new_pos.protection_orders.tp_order_id = _protection_id(existing_tp.get("algoId"))
                        new_pos.protection_orders.tp_client_order_id = existing_tp.get("clientAlgoId") or f"TP_{_short_id(new_pos_id)}"
                        logger.info("Linked to existing exchange TP on adoption", symbol=symbol, algo_id=existing_tp.get("algoId"))
                    else:
                        new_pos.protection_orders.tp_price = tp_price
                        new_pos.protection_orders.tp_client_order_id = f"TP_{_short_id(new_pos_id)}"

                    if existing_stop is None or existing_tp is None:
                        new_pos.protection_orders.status = "PENDING"

                    await portfolio_mgr.add_position(new_pos)

                    adopted_count += 1
                    results["adopted"] = results.get("adopted", 0) + 1
                    protection_status = new_pos.protection_orders.status
                    results["details"].append({
                        "type": "ADOPTED",
                        "symbol": symbol,
                        "position_id": new_pos.position_id,
                        "exchange_qty": exchange_qty,
                        "entry_price": ex_pos["entry_price"],
                        "mark_price": mark_price,
                        "stop_price": sl_price,
                        "tp_price": tp_price,
                        "protection_status": protection_status,
                    })
                    if protection_status == "VERIFIED":
                        logger.info(
                            "Adopted unmanaged position — linked to existing exchange protection",
                            symbol=symbol, position_id=new_pos.position_id,
                            stop_price=sl_price, tp_price=tp_price,
                        )
                    else:
                        logger.info(
                            "Adopted unmanaged position — protection pending audit/repair cycle",
                            symbol=symbol, position_id=new_pos.position_id,
                            stop_price=sl_price, tp_price=tp_price,
                        )
                except Exception as e:
                    logger.error(
                        "Failed to adopt unmanaged position",
                        symbol=symbol,
                        error=str(e),
                    )
                    results["details"].append({
                        "type": "ADOPTION_FAILED",
                        "symbol": symbol,
                        "exchange_qty": exchange_qty,
                        "entry_price": ex_pos["entry_price"],
                        "error": str(e),
                    })

                    try:
                        event = SystemEvent(
                            event_type="UNMANAGED_POSITION_DETECTED",
                            service_name="Reconciler",
                            payload={
                                "symbol": symbol,
                                "exchange_qty": exchange_qty,
                                "entry_price": ex_pos["entry_price"],
                                "leverage": ex_pos.get("leverage"),
                                "margin_type": ex_pos.get("margin_type"),
                                "unrealized_profit": ex_pos.get("unrealized_profit"),
                                "open_orders_on_symbol": symbol in open_order_symbols,
                                "adoption_error": str(e),
                            },
                        )
                        portfolio_mgr._store.append_audit_log(event)
                    except Exception as e2:
                        logger.error(
                            "Failed to append audit log for unmanaged position",
                            symbol=symbol,
                            error=str(e2),
                        )

        # Refresh local_by_symbol to include newly adopted positions
        # so that steps 4c–4g operate on current state instead of the
        # pre-adoption snapshot.
        local_positions = portfolio_mgr.get_live_positions()
        local_by_symbol = {p.symbol: p for p in local_positions}

        # --- Step 4c: MISMATCH — both exist, verify quantities ---
        for symbol, local_pos in local_by_symbol.items():
            ex_pos = exchange_by_symbol.get(symbol)
            if ex_pos is None:
                continue

            exchange_qty = ex_pos["position_amt"]
            local_qty = local_pos.quantity
            qty_delta = abs(exchange_qty - local_qty)

            if qty_delta > DUCKDB_QTY_TOLERANCE:
                if not (math.isfinite(exchange_qty) and exchange_qty > 0):
                    logger.error(
                        "Exchange quantity invalid, refusing to sync",
                        symbol=symbol, position_id=local_pos.position_id,
                        exchange_qty=exchange_qty,
                    )
                    continue
                if local_qty > 0 and exchange_qty / local_qty > 2.0:
                    logger.warning(
                        "Exchange quantity more than 2x local — syncing but flagging anomaly",
                        symbol=symbol, position_id=local_pos.position_id,
                        exchange_qty=exchange_qty, local_qty=local_qty,
                        ratio=exchange_qty / local_qty,
                    )
                logger.warning(
                    "Quantity mismatch detected between exchange and local. "
                    "Synchronizing local to exchange value.",
                    symbol=symbol,
                    position_id=local_pos.position_id,
                    exchange_qty=exchange_qty,
                    local_qty=local_qty,
                    delta=qty_delta,
                )
                local_pos.quantity = exchange_qty
                try:
                    portfolio_mgr._store.save_position(local_pos)
                except Exception as e:
                    logger.error(
                        "Failed to save synchronized position",
                        symbol=symbol,
                        position_id=local_pos.position_id,
                        error=str(e),
                    )
                results["qty_mismatches"] += 1
                results["details"].append({
                    "type": "QUANTITY_MISMATCH",
                    "symbol": symbol,
                    "position_id": local_pos.position_id,
                    "exchange_qty": exchange_qty,
                    "local_qty": local_qty,
                    "delta": qty_delta,
                })

        # --- Step 4d: PROTECTION SELF-HEALING — verify algo protective orders ---
        for symbol, local_pos in local_by_symbol.items():
            ex_pos = exchange_by_symbol.get(symbol)
            if ex_pos is None:
                continue

            algo_ords = algo_by_symbol.get(symbol, [])
            pos_short = _short_id(local_pos.position_id)
            has_stop = any(
                o.get("clientAlgoId", "").startswith(f"SL_{pos_short}")
                for o in algo_ords
            )
            has_tp = any(
                o.get("clientAlgoId", "").startswith(f"TP_{pos_short}")
                for o in algo_ords
            )

            side = "SELL" if local_pos.side == "LONG" else "BUY"
            qty = local_pos.quantity
            price = local_pos.avg_fill_price
            stop_price = local_pos.protection_orders.stop_price
            tp_price = local_pos.protection_orders.tp_price

            if not has_stop and stop_price > 0:
                # Check for ANY existing STOP_MARKET (adopted positions may have non-SL_ prefix)
                existing_any_stop = _find_stop_order(algo_ords, side)
                if existing_any_stop is not None:
                    local_pos.protection_orders.stop_order_id = _protection_id(existing_any_stop.get("algoId"))
                    local_pos.protection_orders.stop_client_order_id = existing_any_stop.get("clientAlgoId")
                    local_pos.protection_orders.stop_price = float(existing_any_stop.get("triggerPrice", stop_price))
                    logger.info(
                        "Linked to existing exchange protection during self-healing",
                        symbol=symbol, position_id=local_pos.position_id,
                        client_algo_id=existing_any_stop.get("clientAlgoId"),
                    )
                    results.setdefault("protection_linked", 0)
                    results["protection_linked"] += 1
                else:
                    if not (math.isfinite(stop_price) and stop_price > 0):
                        logger.error(
                            "Invalid stop price for recreation, skipping",
                            symbol=symbol, position_id=local_pos.position_id,
                            stop_price=stop_price,
                        )
                    else:
                        logger.warning(
                            "Algo stop protection missing on exchange — recreating",
                            symbol=symbol, position_id=local_pos.position_id,
                            stop_price=stop_price,
                        )
                        stop_cid = f"SL_{pos_short}"
                        stop_ok = False
                        for attempt in range(2):
                            try:
                                stop_resp = await client.place_algo_stop(
                                    symbol, side, stop_price, local_pos.position_id,
                                    client_algo_id=stop_cid,
                                    estimated_qty=qty, current_price=price,
                                )
                                await asyncio.sleep(0.3)
                                algo_ords = await client.get_open_algo_orders(symbol)
                                if any(o.get("clientAlgoId") == stop_cid for o in algo_ords):
                                    local_pos.protection_orders.stop_order_id = _protection_id(stop_resp.get("algoId"))
                                    local_pos.protection_orders.stop_client_order_id = stop_cid
                                    stop_ok = True
                                    break
                            except Exception as e:
                                error_str = str(e)
                                if "-4130" in error_str:
                                    try:
                                        algo_ords = await client.get_open_algo_orders(symbol)
                                        existing = _find_stop_order(algo_ords, side)
                                        if existing:
                                            local_pos.protection_orders.stop_order_id = _protection_id(existing.get("algoId"))
                                            local_pos.protection_orders.stop_client_order_id = existing.get("clientAlgoId", stop_cid)
                                            local_pos.protection_orders.stop_price = float(existing.get("triggerPrice", stop_price))
                                            stop_ok = True
                                            logger.info(
                                                "-4130: Linked to existing stop on exchange",
                                                symbol=symbol, algo_id=existing.get("algoId"),
                                            )
                                            break
                                    except Exception:
                                        pass
                                logger.warning(
                                    "Stop recreation attempt failed",
                                    symbol=symbol, attempt=attempt + 1, error=str(e),
                                )
                                await asyncio.sleep(0.5)
                        if stop_ok:
                            logger.info(
                                "Algo stop protection recreated and verified",
                                symbol=symbol, algo_id=local_pos.protection_orders.stop_order_id,
                            )
                            results.setdefault("protection_recovered", 0)
                            results["protection_recovered"] += 1
                            results["details"].append({
                                "type": "STOP_RECOVERED",
                                "symbol": symbol,
                                "position_id": local_pos.position_id,
                            })
                        else:
                            logger.critical(
                                "Failed to recreate algo stop protection",
                                symbol=symbol, position_id=local_pos.position_id,
                            )
                            results["details"].append({
                                "type": "STOP_RECOVERY_FAILED",
                                "symbol": symbol,
                                "position_id": local_pos.position_id,
                            })

            if not has_tp and tp_price > 0:
                if not (math.isfinite(tp_price) and tp_price > 0):
                    logger.error(
                        "Invalid TP price for recreation, skipping",
                        symbol=symbol, position_id=local_pos.position_id,
                        tp_price=tp_price,
                    )
                else:
                    logger.warning(
                        "Algo TP protection missing on exchange — recreating",
                        symbol=symbol, position_id=local_pos.position_id,
                        tp_price=tp_price,
                    )
                    tp_cid = f"TP_{pos_short}"
                    tp_ok = False
                    for attempt in range(2):
                        try:
                            tp_resp = await client.place_algo_tp(
                                symbol, side, tp_price, local_pos.position_id,
                                client_algo_id=tp_cid,
                                estimated_qty=qty, current_price=price,
                            )
                            await asyncio.sleep(0.3)
                            algo_ords = await client.get_open_algo_orders(symbol)
                            if any(o.get("clientAlgoId") == tp_cid for o in algo_ords):
                                local_pos.protection_orders.tp_order_id = tp_resp.get("algoId")
                                local_pos.protection_orders.tp_client_order_id = tp_cid
                                tp_ok = True
                                break
                        except Exception as e:
                            error_str = str(e)
                            if "-4130" in error_str:
                                try:
                                    algo_ords = await client.get_open_algo_orders(symbol)
                                    existing = _find_tp_order(algo_ords, side)
                                    if existing:
                                        local_pos.protection_orders.tp_order_id = _protection_id(existing.get("algoId"))
                                        local_pos.protection_orders.tp_client_order_id = existing.get("clientAlgoId", tp_cid)
                                        local_pos.protection_orders.tp_price = float(existing.get("triggerPrice", tp_price))
                                        tp_ok = True
                                        logger.info(
                                            "-4130: Linked to existing TP on exchange",
                                            symbol=symbol, algo_id=existing.get("algoId"),
                                        )
                                        break
                                except Exception:
                                    pass
                            logger.warning(
                                "TP recreation attempt failed",
                                symbol=symbol, attempt=attempt + 1, error=str(e),
                            )
                            await asyncio.sleep(0.5)
                    if tp_ok:
                        logger.info(
                            "Algo TP protection recreated and verified",
                            symbol=symbol, algo_id=local_pos.protection_orders.tp_order_id,
                        )
                        results.setdefault("protection_recovered", 0)
                        results["protection_recovered"] += 1
                        results["details"].append({
                            "type": "TP_RECOVERED",
                            "symbol": symbol,
                            "position_id": local_pos.position_id,
                        })
                    else:
                        logger.critical(
                            "Failed to recreate algo TP protection",
                            symbol=symbol, position_id=local_pos.position_id,
                        )
                        results["details"].append({
                            "type": "TP_RECOVERY_FAILED",
                            "symbol": symbol,
                            "position_id": local_pos.position_id,
                        })

            if has_stop or has_tp:
                for o in algo_ords:
                    cid = o.get("clientAlgoId", "")
                    if cid.startswith(f"SL_{pos_short}"):
                        local_pos.protection_orders.stop_order_id = _protection_id(o.get("algoId"))
                        local_pos.protection_orders.stop_price = float(o.get("triggerPrice", stop_price))
                    elif cid.startswith(f"TP_{pos_short}"):
                        local_pos.protection_orders.tp_order_id = _protection_id(o.get("algoId"))
                        local_pos.protection_orders.tp_price = float(o.get("triggerPrice", tp_price))

            portfolio_mgr._store.save_position(local_pos)

        # --- Step 4e: CANCEL STALE ALGO PROTECTION — position closed, protection still open ---
        for symbol, ex_pos in exchange_by_symbol.items():
            local_pos = local_by_symbol.get(symbol)
            if local_pos is not None:
                continue
            for o in algo_orders:
                if o.get("symbol") != symbol:
                    continue
                cid = o.get("clientAlgoId", "")
                if cid.startswith("SL_") or cid.startswith("TP_"):
                    logger.warning(
                        "Stale algo protection order for closed position — cancelling",
                        symbol=symbol, algo_id=o.get("algoId"), client_id=cid,
                    )
                    try:
                        await client.cancel_algo_by_client_id(symbol, cid)
                    except Exception as e:
                        logger.error("Failed to cancel stale algo protection", error=str(e))

        # --- Step 4f: RESUME PENDING PROTECTION — positions with protection in PENDING state ---
        for symbol, local_pos in local_by_symbol.items():
            if local_pos.protection_orders.status == "PENDING":
                logger.warning(
                    "Position has PENDING protection — requesting repair",
                    symbol=symbol, position_id=local_pos.position_id,
                )
                pending_payload = {
                    "position_id": local_pos.position_id,
                    "symbol": symbol,
                    "side": local_pos.side,
                    "stop_price": local_pos.protection_orders.stop_price,
                    "tp_price": local_pos.protection_orders.tp_price,
                    "quantity": local_pos.quantity,
                    "execution_id": local_pos.position_id,
                }
                portfolio_mgr._store.append_audit_log(SystemEvent(
                    event_type="PROTECTION_REPAIR_REQUESTED",
                    service_name="Reconciler",
                    payload=pending_payload,
                ))
                try:
                    await portfolio_mgr._event_bus.publish(SystemEvent(
                        event_type="PROTECTION_REPAIR_REQUESTED",
                        service_name="Reconciler",
                        payload=pending_payload,
                    ))
                except Exception as e:
                    logger.error("Failed to publish PROTECTION_REPAIR_REQUESTED", error=str(e))
                results.setdefault("protection_resumed", 0)
                results["protection_resumed"] += 1
                results["details"].append({
                    "type": "PROTECTION_RESUMED",
                    "symbol": symbol,
                    "position_id": local_pos.position_id,
                })

        # --- Step 4g: DETECT LOST PROTECTION — exchange has no matching algo for expected IDs ---
        for symbol, local_pos in local_by_symbol.items():
            if local_pos.execution_mode != "LIVE":
                continue
            if local_pos.protection_orders.status in ("REMOVED", "FAILED"):
                continue
            expected_ids = set()
            if local_pos.protection_orders.stop_client_order_id:
                expected_ids.add(local_pos.protection_orders.stop_client_order_id)
            if local_pos.protection_orders.tp_client_order_id:
                expected_ids.add(local_pos.protection_orders.tp_client_order_id)
            if not expected_ids:
                continue
            algo_ords = algo_by_symbol.get(symbol, [])
            found_ids = {o.get("clientAlgoId") for o in algo_ords if o.get("clientAlgoId")}
            missing = expected_ids - found_ids
            if missing:
                logger.warning(
                    "PROTECTION_LOST_ON_EXCHANGE",
                    symbol=symbol, position_id=local_pos.position_id,
                    missing=list(missing),
                )
                portfolio_mgr._store.append_audit_log(SystemEvent(
                    event_type="PROTECTION_LOST",
                    service_name="Reconciler",
                    payload={
                        "position_id": local_pos.position_id,
                        "symbol": symbol,
                        "missing_ids": list(missing),
                        "reason": "reconciliation_protection_mismatch",
                    },
                ))
                repair_payload = {
                    "position_id": local_pos.position_id,
                    "symbol": symbol,
                    "side": local_pos.side,
                    "stop_price": local_pos.protection_orders.stop_price,
                    "tp_price": local_pos.protection_orders.tp_price,
                    "quantity": local_pos.quantity,
                    "execution_id": local_pos.position_id,
                }
                portfolio_mgr._store.append_audit_log(SystemEvent(
                    event_type="PROTECTION_REPAIR_REQUESTED",
                    service_name="Reconciler",
                    payload=repair_payload,
                ))
                try:
                    await portfolio_mgr._event_bus.publish(SystemEvent(
                        event_type="PROTECTION_REPAIR_REQUESTED",
                        service_name="Reconciler",
                        payload=repair_payload,
                    ))
                except Exception as e:
                    logger.error("Failed to publish PROTECTION_REPAIR_REQUESTED", error=str(e))
                results.setdefault("protection_lost_detected", 0)
                results["protection_lost_detected"] += 1
                results["details"].append({
                    "type": "PROTECTION_LOST",
                    "symbol": symbol,
                    "position_id": local_pos.position_id,
                    "missing": list(missing),
                })

        # --- Step 4h: PROMOTE adopted positions to OPEN for full lifecycle tracking ---
        for symbol, local_pos in local_by_symbol.items():
            if local_pos.lifecycle_state != PositionState.UNMANAGED_ADOPTED:
                continue
            if local_pos.protection_orders.status in ("PENDING", "FAILED", "REMOVED"):
                logger.info(
                    "Adopted position still not fully linked — deferring promotion",
                    symbol=symbol, position_id=local_pos.position_id,
                    status=local_pos.protection_orders.status,
                )
                continue
            local_pos.protection_orders.status = "ACTIVE"
            try:
                await portfolio_mgr.update_position_state(
                    local_pos.position_id,
                    PositionState.OPEN,
                )
                results.setdefault("promoted", 0)
                results["promoted"] += 1
                logger.info(
                    "Adopted position promoted to OPEN",
                    symbol=symbol, position_id=local_pos.position_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to promote adopted position to OPEN",
                    symbol=symbol, position_id=local_pos.position_id, error=str(e),
                )

        logger.info(
            "Reconciliation complete",
            orphaned_closed=results["orphaned_closed"],
            adopted=results["adopted"],
            qty_mismatches=results["qty_mismatches"],
            protection_recovered=results.get("protection_recovered", 0),
            protection_resumed=results.get("protection_resumed", 0),
            protection_lost=results.get("protection_lost_detected", 0),
            promoted=results.get("promoted", 0),
            exchange_positions=results["exchange_positions"],
            local_open_positions=results["local_open_positions"],
            open_orders=results["open_orders"],
            open_algo_orders=results["open_algo_orders"],
        )

        return results
