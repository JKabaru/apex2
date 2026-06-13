from datetime import datetime, timezone

import duckdb
import structlog

logger = structlog.get_logger("correlation_matrix")

DB_PATH = "data/correlation_matrix.duckdb"


class CorrelationMatrixStore:
    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._conn = duckdb.connect(db_path)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS correlation_snapshot (
                timestamp TEXT NOT NULL,
                pair VARCHAR NOT NULL,
                timeframe VARCHAR DEFAULT '1m',
                coefficient REAL,
                dominant_lag INT,
                direction INT,
                p_value REAL,
                n_eff_joint REAL,
                significant BOOLEAN
            )
        """)
        self._conn.execute("ALTER TABLE correlation_snapshot ADD COLUMN IF NOT EXISTS anchor VARCHAR")
        self._conn.execute("ALTER TABLE correlation_snapshot ADD COLUMN IF NOT EXISTS alt VARCHAR")
        self._conn.commit()
        self.prune_old_snapshots()

    def prune_old_snapshots(self, retention_days: int = 7):
        self._conn.execute(
            f"DELETE FROM correlation_snapshot WHERE CAST(timestamp AS TIMESTAMP WITH TIME ZONE) < (CURRENT_TIMESTAMP - INTERVAL '{retention_days} DAYS')"
        )
        self._conn.commit()
        logger.info("Pruned old correlation snapshots", retention_days=retention_days)

    def insert_snapshot(self, results: list[dict]):
        if not results:
            return
        ts = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                ts,
                r.get("pair", "UNKNOWN"),
                r.get("anchor", ""),
                r.get("alt", ""),
                r.get("timeframe", "1m"),
                r.get("coefficient", 0.0),
                r.get("dominant_lag", -1),
                r.get("direction", 0),
                r.get("p_value", 1.0),
                r.get("n_eff_joint", 0.0),
                int(r.get("significant", False)),
            )
            for r in results
        ]
        self._conn.executemany("""
            INSERT INTO correlation_snapshot
                (timestamp, pair, anchor, alt, timeframe, coefficient, dominant_lag,
                 direction, p_value, n_eff_joint, significant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self._conn.commit()
        logger.info("Correlation snapshot inserted", rows=len(rows))

    def get_latest_matrix(self) -> list[dict]:
        cursor = self._conn.execute("""
            SELECT DISTINCT ON (timeframe, pair)
                   timestamp, timeframe, pair, anchor, alt, coefficient, dominant_lag,
                   direction, p_value, n_eff_joint, significant
            FROM correlation_snapshot
            ORDER BY timeframe, pair, timestamp DESC
        """)
        return [
            {
                "timestamp": row[0],
                "timeframe": row[1],
                "pair": row[2],
                "anchor": row[3],
                "alt": row[4],
                "coefficient": row[5],
                "dominant_lag": row[6],
                "direction": row[7],
                "p_value": row[8],
                "n_eff_joint": row[9],
                "significant": bool(row[10]),
            }
            for row in cursor.fetchall()
        ]

    def close(self):
        self._conn.close()
