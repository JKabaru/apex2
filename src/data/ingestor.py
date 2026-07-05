from datetime import datetime, timezone

import duckdb
import structlog

from src.api.binance_client import BinanceClient
from src.engine.output_mode import is_verbose

DB_PATH = "data/ohlcv.duckdb"


class Ingestor:
    def __init__(self, mode: str, binance_client: BinanceClient):
        self.mode = mode
        self.client = binance_client
        self.log = structlog.get_logger("ingestor")
        self._conn = duckdb.connect(DB_PATH)
        self._init_db()

    def _get_conn(self):
        return self._conn

    def _init_db(self):
        conn = self._conn
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_1m (
                    symbol VARCHAR,
                    open_time TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    PRIMARY KEY (symbol, open_time)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gap_log (
                    symbol VARCHAR,
                    gap_start TIMESTAMP,
                    gap_end TIMESTAMP,
                    filled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            self.log.info("Database initialized", path=DB_PATH)
        except Exception:
            pass

    def close(self):
        self._conn.close()

    @staticmethod
    def _ms_to_ts(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

    async def append_candle(self, symbol: str, candle: dict):
        open_time_ms = candle["open_time"]
        close_time_ms = candle["close_time"]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if close_time_ms >= now_ms:
            self.log.warning(
                "Skipping candle that is not yet fully closed",
                symbol=symbol,
                close_time=close_time_ms,
                now=now_ms,
            )
            return

        candle_ts = self._ms_to_ts(open_time_ms)
        conn = self._conn
        try:
            row = conn.execute(
                "SELECT MAX(open_time) FROM ohlcv_1m WHERE symbol = ?",
                [symbol],
            ).fetchone()
            last_ts = row[0] if row and row[0] else None

            if last_ts is not None:
                gap_start_ms = int(last_ts.timestamp() * 1000) + 60000
                gap_end_ms = open_time_ms - 60000

                if gap_start_ms <= gap_end_ms:
                    self.log.info(
                        "Gap detected",
                        symbol=symbol,
                        gap_start=self._ms_to_ts(gap_start_ms),
                        gap_end=self._ms_to_ts(gap_end_ms),
                    )
                    await self._fill_gap(symbol, gap_start_ms, gap_end_ms, conn)

            if is_verbose():
                self.log.info(
                    "Inserting closed candle",
                    symbol=symbol,
                    open_time=candle_ts,
                    open=candle["open"],
                    close=candle["close"],
                )

            conn.execute(
                """
                INSERT INTO ohlcv_1m (symbol, open_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, open_time) DO NOTHING
                """,
                [
                    symbol,
                    candle_ts,
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"],
                ],
            )
            conn.commit()
        except Exception:
            pass

    async def _fill_gap(self, symbol: str, gap_start_ms: int, gap_end_ms: int, conn):
        self.log.info(
            "Filling gap from REST",
            symbol=symbol,
            start_ms=gap_start_ms,
            end_ms=gap_end_ms,
        )

        try:
            candles = await self.client.get_historical_klines(
                symbol=symbol,
                start_time=gap_start_ms,
                end_time=gap_end_ms,
                interval="1m",
            )
        except Exception as e:
            self.log.error(
                "Failed to fetch historical klines for gap fill",
                symbol=symbol,
                error=str(e),
            )
            return

        if not candles:
            self.log.info("No gap candles returned by API", symbol=symbol)
            return

        inserted = 0
        for c in candles:
            c_ts = self._ms_to_ts(c["open_time"])
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            if c["close_time"] >= now_ms:
                continue

            result = conn.execute(
                """
                INSERT INTO ohlcv_1m (symbol, open_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, open_time) DO NOTHING
                """,
                [symbol, c_ts, c["open"], c["high"], c["low"], c["close"], c["volume"]],
            )
            inserted += 1

        conn.execute(
            """
            INSERT INTO gap_log (symbol, gap_start, gap_end)
            VALUES (?, ?, ?)
            """,
            [
                symbol,
                self._ms_to_ts(gap_start_ms),
                self._ms_to_ts(gap_end_ms),
            ],
        )
        conn.commit()

        self.log.info(
            "Gap fill complete",
            symbol=symbol,
            candles_inserted=inserted,
            gap_range=f"{gap_start_ms}-{gap_end_ms}",
        )
