from datetime import datetime, timezone

import duckdb
import structlog

DB_PATH = "data/ohlcv.duckdb"

TIMEFRAMES = [
    ("15m", "INTERVAL 15 MINUTES"),
    ("1h", "INTERVAL 1 HOUR"),
    ("4h", "INTERVAL 4 HOURS"),
]


class Aggregator:
    def __init__(self):
        self.log = structlog.get_logger("aggregator")
        self._init_db()

    def _get_conn(self):
        return duckdb.connect(DB_PATH)

    def _init_db(self):
        conn = self._get_conn()
        try:
            for tf_name, _ in TIMEFRAMES:
                table_name = f"ohlcv_{tf_name}"
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        symbol VARCHAR,
                        bucket TIMESTAMP,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        volume DOUBLE,
                        PRIMARY KEY (symbol, bucket)
                    )
                """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregation_watermark (
                    timeframe VARCHAR PRIMARY KEY,
                    max_bucket TIMESTAMP
                )
            """)
            conn.commit()
            self.log.info("Aggregation tables initialized")
        finally:
            conn.close()

    async def aggregate_timeframes(self):
        conn = self._get_conn()
        try:
            for tf_name, tf_interval in TIMEFRAMES:
                table_name = f"ohlcv_{tf_name}"
                await self._aggregate_single(conn, tf_name, tf_interval, table_name)
        finally:
            conn.close()

    async def _aggregate_single(self, conn, tf_name: str, tf_interval: str, table_name: str):
        row = conn.execute(
            "SELECT max_bucket FROM aggregation_watermark WHERE timeframe = ?",
            [tf_name],
        ).fetchone()

        if row and row[0]:
            watermark = row[0]
            self.log.info("Processing delta aggregation", timeframe=tf_name, watermark=str(watermark))
        else:
            watermark = datetime(2000, 1, 1, tzinfo=timezone.utc)
            self.log.info("No watermark found, processing all data", timeframe=tf_name)

        now = datetime.now(timezone.utc)
        now_rounded = now.replace(second=0, microsecond=0)

        result = conn.execute(f"""
            SELECT
                symbol,
                time_bucket({tf_interval}, open_time) AS bucket,
                argMin(open, open_time) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                argMax(close, open_time) AS close,
                SUM(volume) AS volume
            FROM ohlcv_1m
            WHERE open_time > ?
                AND open_time < ?
            GROUP BY symbol, bucket
        """, [watermark, now_rounded]).fetchall()

        if not result:
            self.log.info("No new data to aggregate", timeframe=tf_name)
            return

        inserted = 0
        for row_data in result:
            symbol, bucket, open_, high, low, close, volume = row_data
            conn.execute(f"""
                INSERT INTO {table_name} (symbol, bucket, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, bucket) DO NOTHING
            """, [symbol, bucket, open_, high, low, close, volume])
            inserted += 1

        max_bucket = conn.execute(f"""
            SELECT MAX(bucket) FROM {table_name}
        """).fetchone()[0]

        conn.execute("""
            INSERT INTO aggregation_watermark (timeframe, max_bucket)
            VALUES (?, ?)
            ON CONFLICT (timeframe) DO UPDATE SET max_bucket = EXCLUDED.max_bucket
        """, [tf_name, max_bucket])

        conn.commit()

        self.log.info(
            "Aggregation complete",
            timeframe=tf_name,
            rows_inserted=inserted,
            max_bucket=str(max_bucket),
        )
