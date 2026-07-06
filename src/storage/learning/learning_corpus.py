from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import duckdb
import structlog

from src.models.learning.trade_experience import LearningManifest
from src.models.learning.corpus_metadata import CorpusMetadata
from src.retrieval.models import CorpusDiagnostics, RetrievalRecord
from src.retrieval.projection import CorpusProjection

logger = structlog.get_logger("learning_corpus")

CORPUS_DB = "data/experience_corpus.duckdb"


class LearningCorpus:
    """Append-only immutable storage for LearningManifest objects.
    Only complete manifests are persisted — never partial artifacts.
    The database is a separate DuckDB file, completely independent
    of the trading engine's portfolio database."""

    def __init__(self, db_path: str = CORPUS_DB):
        try:
            self._conn = duckdb.connect(db_path)
        except duckdb.InternalException:
            wal_path = db_path + ".wal"
            if os.path.exists(wal_path):
                os.remove(wal_path)
                self._conn = duckdb.connect(db_path)
            else:
                raise
        logger.info("LearningCorpus initialized", db_path=db_path)

    def create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                experience_id VARCHAR PRIMARY KEY,
                position_id VARCHAR NOT NULL,
                manifest_json JSON NOT NULL,
                hash VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL DEFAULT '2.0',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_corpus_position_id
            ON experiences (position_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_corpus_created_at
            ON experiences (created_at)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS corpus_meta (
                key VARCHAR PRIMARY KEY,
                value JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Learning corpus schema initialized (v2.0)")

    def save(self, manifest: LearningManifest) -> None:
        manifest_dict = manifest.model_dump(mode="json")
        self._conn.execute("""
            INSERT INTO experiences (experience_id, position_id, manifest_json, hash, schema_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (experience_id) DO NOTHING
        """, [
            manifest.experience_id,
            manifest.position_id,
            json.dumps(manifest_dict),
            manifest.hash,
            manifest.schema_version,
            datetime.utcnow(),
        ])

    def load(self, experience_id: str) -> Optional[LearningManifest]:
        row = self._conn.execute(
            "SELECT manifest_json FROM experiences WHERE experience_id = ?",
            [experience_id],
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def find_by_position_id(self, position_id: str) -> Optional[LearningManifest]:
        row = self._conn.execute(
            "SELECT manifest_json FROM experiences WHERE position_id = ? LIMIT 1",
            [position_id],
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def load_batch(
        self, limit: int = 100, offset: int = 0
    ) -> list[LearningManifest]:
        rows = self._conn.execute(
            "SELECT manifest_json FROM experiences "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
        return [self._deserialize(r[0]) for r in rows]

    def exists(self, experience_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM experiences WHERE experience_id = ?",
            [experience_id],
        ).fetchone()
        return row is not None

    def statistics(self) -> dict[str, Any]:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()[0]
        return {
            "total_experiences": total,
        }

    # ── Corpus Metadata ──

    def save_corpus_metadata(self, metadata: CorpusMetadata) -> None:
        raw = json.dumps(metadata.model_dump(mode="json"))
        self._conn.execute("""
            INSERT INTO corpus_meta (key, value, created_at)
            VALUES ('corpus_metadata', ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, created_at = EXCLUDED.created_at
        """, [raw, datetime.utcnow()])

    def get_corpus_metadata(self) -> Optional[CorpusMetadata]:
        row = self._conn.execute(
            "SELECT value FROM corpus_meta WHERE key = 'corpus_metadata'"
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return CorpusMetadata(**data)

    @staticmethod
    def _deserialize(raw: str) -> LearningManifest:
        data = json.loads(raw)
        return LearningManifest(**data)

    # ── Retrieval View ──

    def get_corpus_view(
        self,
        limit: int = 500,
        filters: Optional[dict] = None,
    ) -> list[RetrievalRecord]:
        rows = self._conn.execute(
            "SELECT manifest_json FROM experiences ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()

        projector = CorpusProjection()
        records = [projector.project_manifest(self._deserialize(r[0])) for r in rows]

        if filters:
            records = self._apply_retrieval_filters(records, filters)

        return records

    @staticmethod
    def _apply_retrieval_filters(
        records: list[RetrievalRecord],
        filters: dict,
    ) -> list[RetrievalRecord]:
        result = list(records)
        if "symbol" in filters and filters["symbol"] is not None:
            val = filters["symbol"]
            result = [r for r in result if r.symbol == val]
        if "timeframe" in filters and filters["timeframe"] is not None:
            val = filters["timeframe"]
            result = [r for r in result if r.timeframe == val]
        if "opportunity_id" in filters and filters["opportunity_id"] is not None:
            val = filters["opportunity_id"]
            result = [r for r in result if r.opportunity_id == val]
        if "market_state_hash" in filters and filters["market_state_hash"] is not None:
            val = filters["market_state_hash"]
            result = [r for r in result if r.market_state_hash == val]
        if "trend_regime" in filters and filters["trend_regime"] is not None:
            val = filters["trend_regime"]
            result = [r for r in result if r.trend_regime == val]
        if "volatility_regime" in filters and filters["volatility_regime"] is not None:
            val = filters["volatility_regime"]
            result = [r for r in result if r.volatility_regime == val]
        if "correlation_regime" in filters and filters["correlation_regime"] is not None:
            val = filters["correlation_regime"]
            result = [r for r in result if r.correlation_regime == val]
        if "min_integrity" in filters and filters["min_integrity"] is not None:
            val = int(filters["min_integrity"])
            result = [r for r in result if r.integrity_score >= val]
        return result

    # ── Diagnostics ──

    def get_diagnostics(self) -> CorpusDiagnostics:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()[0]

        schema_version_rows = self._conn.execute(
            "SELECT schema_version, COUNT(*) AS cnt FROM experiences GROUP BY schema_version ORDER BY cnt DESC"
        ).fetchall()
        schema_versions = {row[0]: row[1] for row in schema_version_rows}

        projector = CorpusProjection()
        records = self.get_corpus_view(limit=10000)

        symbol_dist: dict[str, int] = {}
        timeframe_dist: dict[str, int] = {}
        integrity_scores: list[int] = []
        pipeline_version_dist: dict[str, int] = {}
        catalog_hash_dist: dict[str, int] = {}

        regime_trend: dict[str, int] = {}
        regime_vol: dict[str, int] = {}
        regime_corr: dict[str, int] = {}

        metric_fields = [
            "normalized_entry_atr_multiple",
            "normalized_exit_atr_multiple",
            "pnl_atr_multiple",
            "mfe_atr_multiple",
            "mae_atr_multiple",
            "entry_rsi_percentile",
            "entry_volatility_percentile",
            "holding_duration_minutes",
            "bars_held",
            "total_slippage_bps",
            "total_fees_bps",
            "realized_rr",
            "initial_risk_atr_multiple",
        ]
        missing_counts: dict[str, int] = {f: 0 for f in metric_fields}
        total_count = len(records)

        for r in records:
            symbol_dist[r.symbol] = symbol_dist.get(r.symbol, 0) + 1
            timeframe_dist[r.timeframe] = timeframe_dist.get(r.timeframe, 0) + 1
            integrity_scores.append(r.integrity_score)

            pipeline_version_dist[r.pipeline_version] = pipeline_version_dist.get(r.pipeline_version, 0) + 1
            catalog_hash_dist[r.hash[:12]] = catalog_hash_dist.get(r.hash[:12], 0) + 1

            if r.trend_regime is not None:
                regime_trend[r.trend_regime] = regime_trend.get(r.trend_regime, 0) + 1
            else:
                regime_trend["__missing__"] = regime_trend.get("__missing__", 0) + 1

            if r.volatility_regime is not None:
                regime_vol[r.volatility_regime] = regime_vol.get(r.volatility_regime, 0) + 1
            else:
                regime_vol["__missing__"] = regime_vol.get("__missing__", 0) + 1

            if r.correlation_regime is not None:
                regime_corr[r.correlation_regime] = regime_corr.get(r.correlation_regime, 0) + 1
            else:
                regime_corr["__missing__"] = regime_corr.get("__missing__", 0) + 1

            for fname in metric_fields:
                if getattr(r, fname, None) is None:
                    missing_counts[fname] += 1

        avg_integrity = round(sum(integrity_scores) / len(integrity_scores), 2) if integrity_scores else 0.0
        missing_pct = {
            fname: round(count / total_count * 100, 1) if total_count > 0 else 0.0
            for fname, count in missing_counts.items()
        }

        return CorpusDiagnostics(
            total_experiences=total,
            avg_integrity_score=avg_integrity,
            schema_version_distribution=schema_versions,
            pipeline_version_distribution=pipeline_version_dist,
            catalog_hash_distribution=catalog_hash_dist,
            missing_feature_percentages=missing_pct,
            regime_distributions={
                "trend": regime_trend,
                "volatility": regime_vol,
                "correlation": regime_corr,
            },
            symbol_distribution=dict(sorted(symbol_dist.items(), key=lambda x: -x[1])),
            timeframe_distribution=dict(sorted(timeframe_dist.items(), key=lambda x: -x[1])),
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
