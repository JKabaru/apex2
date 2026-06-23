from __future__ import annotations

import uuid

import structlog

from src.api.binance_client import BinanceClient
from src.core.models import Position, PositionState, SystemEvent
from src.services.portfolio_manager import PortfolioManager

logger = structlog.get_logger("reconciler")

DUCKDB_QTY_TOLERANCE = 1e-8


class Reconciler:
    @staticmethod
    async def reconcile(
        portfolio_mgr: PortfolioManager,
        client: BinanceClient,
    ) -> dict:
        logger.info("Starting exchange-to-local position reconciliation")

        # --- Step 1: Query exchange for open positions ---
        exchange_positions = await client.get_open_positions()
        exchange_by_symbol: dict[str, dict] = {
            p["symbol"]: p for p in exchange_positions
        }

        # --- Step 2: Query exchange for open orders ---
        exchange_orders = await client.get_open_orders()
        open_order_symbols = {o.get("symbol") for o in exchange_orders if o.get("symbol")}

        # --- Step 3: Query local open positions ---
        local_positions = portfolio_mgr.get_open_positions()
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
                        PositionState.CLOSED,
                        exit_reason="ORPHANED_RECONCILIATION",
                    )
                    results["orphaned_closed"] += 1
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
        for symbol, ex_pos in exchange_by_symbol.items():
            local_pos = local_by_symbol.get(symbol)
            if local_pos is None:
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
                    new_pos = Position(
                        position_id=str(uuid.uuid4()),
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
                    await portfolio_mgr.add_position(new_pos)

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

                    # Fallback: emit event for manual review
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

        logger.info(
            "Reconciliation complete",
            orphaned_closed=results["orphaned_closed"],
            adopted=results["adopted"],
            qty_mismatches=results["qty_mismatches"],
            exchange_positions=results["exchange_positions"],
            local_open_positions=results["local_open_positions"],
            open_orders=results["open_orders"],
        )

        return results
