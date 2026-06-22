import structlog

logger = structlog.get_logger("reconciler")


async def reconcile_on_startup(binance_client, trade_manager):
    logger.info("Starting startup reconciliation...")
    try:
        local_trades = await trade_manager.get_open_trades()
        exchange_positions = await binance_client.get_open_positions()

        local_by_symbol = {}
        for trade in local_trades:
            symbol = trade["symbol"]
            local_by_symbol.setdefault(symbol, []).append(trade)

        exchange_by_symbol = {pos["symbol"]: pos for pos in exchange_positions}

        # Scenario 1 & 3: Local open trades
        for symbol, trades in local_by_symbol.items():
            exchange_pos = exchange_by_symbol.get(symbol)

            if not exchange_pos:
                # Scenario 1: Local OPEN, Exchange CLOSED -> Mark local as ORPHANED
                for trade in trades:
                    logger.warning(
                        "Local trade is open but exchange position is closed. Marking as ORPHANED.",
                        trade_id=trade["trade_id"],
                        symbol=symbol,
                    )
                    await trade_manager.close_trade(
                        trade_id=trade["trade_id"],
                        exit_price=trade["entry_price"],
                        fees=0.0,
                        pnl=0.0,
                        reason="ORPHANED",
                    )
            else:
                # Scenario 3: Both OPEN -> Do nothing, monitor will track it
                logger.info(
                    "Local trade matches open exchange position. No action needed.",
                    symbol=symbol,
                    exchange_amt=exchange_pos["position_amt"],
                    local_count=len(trades),
                )

        # Scenario 2: Exchange OPEN, Local MISSING -> Force close on exchange
        for symbol, pos in exchange_by_symbol.items():
            if symbol not in local_by_symbol:
                logger.warning(
                    "Exchange position is open but local trade is missing. Force closing exchange position.",
                    symbol=symbol,
                    position_amt=pos["position_amt"],
                )
                try:
                    await binance_client.force_close_position(symbol, pos["position_amt"])
                except Exception as e:
                    logger.error(
                        "Failed to force close orphaned exchange position",
                        symbol=symbol,
                        error=str(e),
                    )

        logger.info("Startup reconciliation complete.")
    except Exception as e:
        logger.error("Error during startup reconciliation", error=str(e))
        # Do not raise to prevent bot crash, but log it heavily
