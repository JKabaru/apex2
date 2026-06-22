import uuid
from datetime import datetime, timezone

import duckdb
import structlog

TRADES_DB = "data/active_trades.duckdb"


class TradeManager:
    def __init__(self, db_path: str = TRADES_DB):
        self._conn = duckdb.connect(db_path)
        self.log = structlog.get_logger("trade_manager")

    async def create_trades_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                entry_time TEXT,
                position_size REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'OPEN',
                exit_price REAL,
                exit_time TEXT,
                fees REAL,
                realized_pnl REAL,
                exit_reason TEXT
            )
        """)
        self.log.info("Trades table ensured")

    async def cleanup_broken_trades(self):
        result = self._conn.execute(
            "UPDATE trades SET status='FAILED', exit_reason='INVALID_SL_TP' "
            "WHERE status='OPEN' AND (stop_loss <= 0 OR take_profit <= 0)"
        )
        count = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE exit_reason='INVALID_SL_TP'"
        ).fetchone()[0]
        if count:
            self.log.info("Cleaned up broken trades", count=count)

    async def open_trade(self, trade_data: dict) -> str:
        trade_id = trade_data.get("trade_id", uuid.uuid4().hex)
        self._conn.execute(
            """
            INSERT INTO trades (
                trade_id, symbol, side, entry_price, entry_time,
                position_size, stop_loss, take_profit, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            [
                trade_id,
                trade_data["symbol"],
                trade_data["side"],
                trade_data["entry_price"],
                trade_data.get("entry_time", datetime.now(timezone.utc).isoformat()),
                trade_data["position_size"],
                trade_data["stop_loss"],
                trade_data["take_profit"],
            ],
        )
        self.log.info("Trade opened", trade_id=trade_id, symbol=trade_data["symbol"], side=trade_data["side"])
        return trade_id

    async def close_trade(self, trade_id: str, exit_price: float, fees: float, pnl: float, reason: str):
        self._conn.execute(
            """
            UPDATE trades
            SET status = 'CLOSED',
                exit_price = ?,
                exit_time = ?,
                fees = ?,
                realized_pnl = ?,
                exit_reason = ?
            WHERE trade_id = ?
            """,
            [exit_price, datetime.now(timezone.utc).isoformat(), fees, pnl, reason, trade_id],
        )
        self.log.info("Trade closed", trade_id=trade_id, pnl=pnl, reason=reason)

    async def get_open_trades(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in rows]

    def close(self):
        self._conn.close()
