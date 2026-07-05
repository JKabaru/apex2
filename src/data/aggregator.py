import threading
from datetime import datetime, timezone

import duckdb
import structlog

from src.engine.output_mode import is_verbose

DB_PATH = "data/ohlcv.duckdb"


class Aggregator:
    def __init__(self, max_timeframe_m: int = 1440, batch_size: int = 100):
        self.max_timeframe_m = max_timeframe_m
        self._batch_size = batch_size
        self.log = structlog.get_logger("aggregator")
        self._conn = duckdb.connect(DB_PATH)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        return self._conn

    def close(self):
        with self._lock:
            self._conn.close()

    def _init_db(self):
        conn = self._conn
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_agg (
                    tf_minutes INTEGER NOT NULL,
                    symbol VARCHAR NOT NULL,
                    bucket TIMESTAMP NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    PRIMARY KEY (tf_minutes, symbol, bucket)
                )
            """)
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_open_time
                    ON ohlcv_1m (open_time)
                """)
            except Exception:
                pass
            conn.execute("DROP TABLE IF EXISTS aggregation_watermark")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregation_watermark (
                    tf_minutes INTEGER PRIMARY KEY,
                    max_bucket TIMESTAMP
                )
            """)
            conn.commit()
            self.log.info("Aggregation tables initialized", max_timeframe_m=self.max_timeframe_m)
        except Exception:
            self.log.error("Failed to initialize aggregation tables", exc_info=True)

    def aggregate_timeframes(self):
        with self._lock:
            conn = self._conn
            try:
                row = conn.execute("SELECT MAX(open_time) FROM ohlcv_1m").fetchone()
                latest_1m = row[0] if row else None
                if latest_1m is None:
                    return

                wm_rows = conn.execute(
                    "SELECT tf_minutes, max_bucket FROM aggregation_watermark"
                ).fetchall()
                watermarks = {r[0]: r[1] for r in wm_rows}

                processed = 0
                for tf_minutes in range(2, self.max_timeframe_m + 1):
                    existing = watermarks.get(tf_minutes)
                    if existing is not None and existing >= latest_1m:
                        continue
                    self._aggregate_single(conn, tf_minutes, latest_1m)
                    processed += 1
                    if processed >= self._batch_size:
                        self.log.info("Batch limit reached", processed=processed)
                        break
            except Exception:
                self.log.error("Aggregation cycle failed", exc_info=True)

    def _aggregate_single(self, conn, tf_minutes: int, latest_1m: datetime):
        try:
            row = conn.execute(
                "SELECT max_bucket FROM aggregation_watermark WHERE tf_minutes = ?",
                [tf_minutes],
            ).fetchone()

            watermark = row[0] if (row and row[0]) else datetime(2000, 1, 1)
            watermark = watermark.replace(tzinfo=None)
            latest_naive = latest_1m.replace(tzinfo=None) if hasattr(latest_1m, 'tzinfo') and latest_1m.tzinfo else latest_1m
            if watermark >= latest_naive:
                return

            interval_str = f"INTERVAL '{tf_minutes} MINUTES'"
            result = conn.execute(f"""
                SELECT
                    symbol,
                    time_bucket({interval_str}, open_time) AS bucket,
                    argMin(open, open_time) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    argMax(close, open_time) AS close,
                    SUM(volume) AS volume
                FROM ohlcv_1m
                WHERE open_time > ? AND open_time < ?
                GROUP BY symbol, bucket
            """, [watermark, latest_1m]).fetchall()

            if not result:
                return

            batch_size = 10000
            rows_batch = []
            for row_data in result:
                symbol, bucket, open_, high, low, close, volume = row_data
                rows_batch.append([tf_minutes, symbol, bucket, open_, high, low, close, volume])

            inserted = 0
            for i in range(0, len(rows_batch), batch_size):
                batch = rows_batch[i:i + batch_size]
                conn.executemany("""
                    INSERT INTO ohlcv_agg (tf_minutes, symbol, bucket, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (tf_minutes, symbol, bucket) DO NOTHING
                """, batch)
                inserted += len(batch)

            res = conn.execute(
                "SELECT MAX(bucket) FROM ohlcv_agg WHERE tf_minutes = ?",
                [tf_minutes],
            ).fetchone()
            max_bucket = res[0] if res and res[0] is not None else 0

            conn.execute("""
                INSERT INTO aggregation_watermark (tf_minutes, max_bucket)
                VALUES (?, ?)
                ON CONFLICT (tf_minutes) DO UPDATE SET max_bucket = EXCLUDED.max_bucket
            """, [tf_minutes, max_bucket])

            conn.commit()

            if inserted and is_verbose():
                self.log.info(
                    "Aggregation complete",
                    tf_minutes=tf_minutes,
                    rows_inserted=inserted,
                    max_bucket=str(max_bucket),
                )
        except Exception:
            self.log.error("Aggregation failed for timeframe", tf_minutes=tf_minutes, exc_info=True)

    def update_timeframe(self, payload: dict, tf_minutes: int):
        with self._lock:
            conn = self._conn
            try:
                open_time_ms = payload["open_time"]
                bucket_ms = tf_minutes * 60_000
                bucket_start_ms = (open_time_ms // bucket_ms) * bucket_ms

                symbol = payload["symbol"]
                bucket_start = datetime.fromtimestamp(
                    bucket_start_ms / 1000.0, tz=timezone.utc
                ).replace(tzinfo=None)
                bucket_end = datetime.fromtimestamp(
                    (bucket_start_ms + bucket_ms) / 1000.0, tz=timezone.utc
                ).replace(tzinfo=None)

                row = conn.execute(
                    "SELECT max_bucket FROM aggregation_watermark WHERE tf_minutes = ?",
                    [tf_minutes],
                ).fetchone()
                watermark = row[0] if (row and row[0]) else datetime(2000, 1, 1)
                if watermark >= bucket_start:
                    return

                result = conn.execute("""
                    SELECT
                        argMin(open, open_time) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        argMax(close, open_time) AS close,
                        SUM(volume) AS volume
                    FROM ohlcv_1m
                    WHERE symbol = ? AND open_time >= ? AND open_time < ?
                """, [symbol, bucket_start, bucket_end]).fetchone()

                if not result or result[0] is None:
                    return

                open_, high, low, close, volume = result

                conn.execute("""
                    INSERT INTO ohlcv_agg (tf_minutes, symbol, bucket, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (tf_minutes, symbol, bucket) DO NOTHING
                """, [tf_minutes, symbol, bucket_start, open_, high, low, close, volume])

                conn.execute("""
                    INSERT INTO aggregation_watermark (tf_minutes, max_bucket)
                    VALUES (?, ?)
                    ON CONFLICT (tf_minutes) DO UPDATE SET max_bucket = EXCLUDED.max_bucket
                """, [tf_minutes, bucket_start])

                conn.commit()
            except Exception:
                self.log.error("Update timeframe failed", tf_minutes=tf_minutes, exc_info=True)
