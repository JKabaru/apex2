from __future__ import annotations

import os

import structlog

from src.api.binance_client import BinanceClient

logger = structlog.get_logger("reset_service")

# Files always wiped on emergency reset
_WIPE_FILES = [
    "data/portfolio.duckdb",
    "data/ohlcv.duckdb",
    "data/agent_params.json",
]

# Files that must NEVER be wiped (learned memory / user config)
_PRESERVE_FILES = [
    "data/configuration_profiles.duckdb",
    "data/experience_corpus.duckdb",
]


class EmergencyResetService:

    @staticmethod
    async def full_data_reset(
        binance_client: BinanceClient,
        log,
    ) -> None:
        """Emergency reset: liquidate exchange positions, wipe stale state files.
        Preserves config, keys, and all memory/knowledge databases."""
        liquidated = 0
        liquidate_errors = 0

        # ── Close all exchange positions ──
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

        # ── Wipe stale state files (keep memory intact) ──
        wiped = []
        kept = []
        for path in _WIPE_FILES:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    wiped.append(path)
                # Also remove WAL files
                wal = path + ".wal"
                if os.path.exists(wal):
                    os.remove(wal)
            except Exception as e:
                log.error("Failed to wipe file", path=path, error=str(e))

        for path in _PRESERVE_FILES:
            if os.path.exists(path):
                kept.append(path)

        # Also discover any workspace DBs (must preserve)
        config_db_path = "data/configuration_profiles.duckdb"
        if os.path.exists(config_db_path):
            try:
                import duckdb
                tmp = duckdb.connect(config_db_path)
                rows = tmp.execute(
                    "SELECT db_path FROM memory_workspaces"
                ).fetchall()
                for row in rows:
                    ws_path = row[0]
                    if os.path.exists(ws_path) and ws_path not in kept:
                        kept.append(ws_path)
                tmp.close()
            except Exception:
                pass

        log.warning(
            "Emergency reset complete",
            liquidated=liquidated,
            liquidate_errors=liquidate_errors,
            files_wiped=len(wiped),
            files_preserved=len(kept),
            _force_log=True,
        )
