from __future__ import annotations

import uuid

import structlog

from src.api.binance_client import BinanceClient
from src.core.models import Position, PositionState, SystemEvent
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("reconciler")

DUCKDB_QTY_TOLERANCE = 1e-8


def _short_id(position_id: str) -> str:
    return position_id.replace("-", "")[:16]


class Reconciler:
    @staticmethod
    async def sync_missing_positions_from_exchange(
        client: BinanceClient,
        portfolio_mgr: PortfolioManager,
    ) -> None:
        """Mid-run adoption: sync any exchange positions missing from local DB."""
        exchange_positions = await client.get_open_positions()
        local_positions = portfolio_mgr.get_live_open_positions()
        local_by_symbol = {p.symbol: p for p in local_positions}

        for ex_pos in exchange_positions:
            symbol = ex_pos["symbol"]
            if symbol in local_by_symbol:
                continue

            exchange_qty = ex_pos["position_amt"]
            mark_price = ex_pos.get("mark_price", ex_pos["entry_price"])
            side = "LONG" if exchange_qty > 0 else "SHORT"
            new_pos_id = str(uuid.uuid4())

            logger.info(
                "Adopting missing exchange position into local DB",
                symbol=symbol, side=side, quantity=abs(exchange_qty),
                entry_price=ex_pos["entry_price"],
            )

            # Check for existing algo protection on exchange
            existing_stop = None
            try:
                algo_orders = await client.get_open_algo_orders(symbol)
                for o in algo_orders:
                    if o.get("type") == "STOP_MARKET":
                        existing_stop = o
                        break
            except Exception:
                logger.warning("Failed to query existing algo orders", symbol=symbol)

            if existing_stop is not None:
                existing_price = float(existing_stop.get("triggerPrice", 0.0))
                existing_cid = existing_stop.get("clientAlgoId", "")
                existing_aid = existing_stop.get("algoId", "")

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
                await portfolio_mgr.add_position(new_pos)
                logger.info(
                    "Adopted position and linked to existing exchange protection",
                    symbol=symbol, position_id=new_pos_id,
                    client_algo_id=existing_cid, stop_price=existing_price,
                )
            else:
                emergency_sl = mark_price * 0.985 if side == "LONG" else mark_price * 1.015
                emergency_side = "SELL" if side == "LONG" else "BUY"
                protection_ok = True

                try:
                    stop_resp = await client.place_algo_stop(
                        symbol, emergency_side, emergency_sl,
                        new_pos_id,
                        estimated_qty=abs(exchange_qty), current_price=mark_price,
                    )
                    sl_price = emergency_sl
                    sl_aid = stop_resp.get("algoId")
                    sl_cid = f"SL_{_short_id(new_pos_id)}"
                except Exception as e:
                    logger.error(
                        "Failed to place emergency stop for adopted position — "
                        "saving position with FAILED protection status",
                        symbol=symbol, error=str(e),
                    )
                    protection_ok = False
                    sl_price = emergency_sl
                    sl_aid = None
                    sl_cid = None

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
                    initial_take_profit=0.0,
                    current_target=0.0,
                    lifecycle_state=PositionState.UNMANAGED_ADOPTED,
                )
                new_pos.protection_orders.stop_price = sl_price
                new_pos.protection_orders.stop_order_id = sl_aid
                new_pos.protection_orders.stop_client_order_id = sl_cid
                new_pos.protection_orders.status = "ACTIVE" if protection_ok else "FAILED"
                await portfolio_mgr.add_position(new_pos)

                if protection_ok:
                    logger.info(
                        "Adopted position and placed new emergency protection",
                        symbol=symbol, position_id=new_pos_id,
                        stop_price=sl_price,
                    )
                else:
                    logger.warning(
                        "Adopted position without exchange protection — "
                        "retry protection on next cycle",
                        symbol=symbol, position_id=new_pos_id,
                    )

    @staticmethod
    async def reconcile(
        portfolio_mgr: PortfolioManager,
        client: BinanceClient,
        max_positions: int = 3,
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
                emergency_sl = mark_price * 0.985 if side == "LONG" else mark_price * 1.015

                logger.warning(
                    "Exchange position detected with no local record. "
                    "Adopting with emergency stop loss.",
                    symbol=symbol,
                    exchange_qty=exchange_qty,
                    entry_price=ex_pos["entry_price"],
                    mark_price=mark_price,
                    emergency_sl=emergency_sl,
                )

                try:
                    new_pos_id = str(uuid.uuid4())
                    emergency_side = "SELL" if side == "LONG" else "BUY"
                    stop_resp = await client.place_algo_stop(
                        symbol, emergency_side, emergency_sl,
                        new_pos_id,
                        estimated_qty=abs(exchange_qty), current_price=mark_price,
                    )
                    new_pos = Position(
                        position_id=new_pos_id,
                        symbol=symbol,
                        side=side,
                        quantity=abs(exchange_qty),
                        avg_fill_price=mark_price,
                        entry_thesis="UNMANAGED_POSITION_ADOPTED_FOR_RISK_PROTECTION",
                        anchor_symbol=symbol,
                        initial_stop_loss=emergency_sl,
                        current_stop=emergency_sl,
                        initial_take_profit=0.0,
                        current_target=0.0,
                        lifecycle_state=PositionState.UNMANAGED_ADOPTED,
                    )
                    new_pos.protection_orders.stop_price = emergency_sl
                    new_pos.protection_orders.stop_order_id = stop_resp.get("algoId")
                    new_pos.protection_orders.stop_client_order_id = f"SL_{_short_id(new_pos_id)}"
                    await portfolio_mgr.add_position(new_pos)

                    adopted_count += 1
                    results["adopted"] = results.get("adopted", 0) + 1
                    results["details"].append({
                        "type": "ADOPTED",
                        "symbol": symbol,
                        "position_id": new_pos.position_id,
                        "exchange_qty": exchange_qty,
                        "entry_price": ex_pos["entry_price"],
                        "mark_price": mark_price,
                        "emergency_sl": emergency_sl,
                        "leverage": ex_pos.get("leverage"),
                        "margin_type": ex_pos.get("margin_type"),
                        "unrealized_profit": ex_pos.get("unrealized_profit"),
                    })
                    logger.info(
                        "Adopted unmanaged position with emergency SL. Now under active monitoring.",
                        symbol=symbol,
                        position_id=new_pos.position_id,
                        emergency_sl=emergency_sl,
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

        # --- Step 4c: MISMATCH — both exist, verify quantities ---
        for symbol, local_pos in local_by_symbol.items():
            ex_pos = exchange_by_symbol.get(symbol)
            if ex_pos is None:
                continue

            exchange_qty = ex_pos["position_amt"]
            local_qty = local_pos.quantity
            qty_delta = abs(exchange_qty - local_qty)

            if qty_delta > DUCKDB_QTY_TOLERANCE:
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
                logger.warning(
                    "Algo stop protection missing on exchange — recreating",
                    symbol=symbol, position_id=local_pos.position_id,
                    stop_price=stop_price,
                )
                try:
                    stop_resp = await client.place_algo_stop(
                        symbol, side, stop_price, local_pos.position_id,
                        estimated_qty=qty, current_price=price,
                    )
                    local_pos.protection_orders.stop_order_id = stop_resp.get("algoId")
                    local_pos.protection_orders.stop_client_order_id = f"SL_{pos_short}"
                    logger.info(
                        "Algo stop protection recreated",
                        symbol=symbol, algo_id=stop_resp.get("algoId"),
                    )
                    results.setdefault("protection_recovered", 0)
                    results["protection_recovered"] += 1
                    results["details"].append({
                        "type": "STOP_RECOVERED",
                        "symbol": symbol,
                        "position_id": local_pos.position_id,
                    })
                except Exception as e:
                    logger.critical(
                        "Failed to recreate algo stop protection",
                        symbol=symbol, position_id=local_pos.position_id, error=str(e),
                    )
                    results["details"].append({
                        "type": "STOP_RECOVERY_FAILED",
                        "symbol": symbol,
                        "position_id": local_pos.position_id,
                        "error": str(e),
                    })

            if not has_tp and tp_price > 0:
                logger.warning(
                    "Algo TP protection missing on exchange — recreating",
                    symbol=symbol, position_id=local_pos.position_id,
                    tp_price=tp_price,
                )
                try:
                    tp_resp = await client.place_algo_tp(
                        symbol, side, tp_price, local_pos.position_id,
                        estimated_qty=qty, current_price=price,
                    )
                    local_pos.protection_orders.tp_order_id = tp_resp.get("algoId")
                    local_pos.protection_orders.tp_client_order_id = f"TP_{pos_short}"
                    logger.info(
                        "Algo TP protection recreated",
                        symbol=symbol, algo_id=tp_resp.get("algoId"),
                    )
                    results.setdefault("protection_recovered", 0)
                    results["protection_recovered"] += 1
                    results["details"].append({
                        "type": "TP_RECOVERED",
                        "symbol": symbol,
                        "position_id": local_pos.position_id,
                    })
                except Exception as e:
                    logger.critical(
                        "Failed to recreate algo TP protection",
                        symbol=symbol, position_id=local_pos.position_id, error=str(e),
                    )
                    results["details"].append({
                        "type": "TP_RECOVERY_FAILED",
                        "symbol": symbol,
                        "position_id": local_pos.position_id,
                        "error": str(e),
                    })

            if has_stop or has_tp:
                for o in algo_ords:
                    cid = o.get("clientAlgoId", "")
                    if cid.startswith(f"SL_{pos_short}"):
                        local_pos.protection_orders.stop_order_id = o.get("algoId")
                        local_pos.protection_orders.stop_price = float(o.get("triggerPrice", stop_price))
                    elif cid.startswith(f"TP_{pos_short}"):
                        local_pos.protection_orders.tp_order_id = o.get("algoId")
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

        logger.info(
            "Reconciliation complete",
            orphaned_closed=results["orphaned_closed"],
            adopted=results["adopted"],
            qty_mismatches=results["qty_mismatches"],
            protection_recovered=results.get("protection_recovered", 0),
            exchange_positions=results["exchange_positions"],
            local_open_positions=results["local_open_positions"],
            open_orders=results["open_orders"],
            open_algo_orders=results["open_algo_orders"],
        )

        return results
