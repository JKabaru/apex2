from __future__ import annotations

import json

import duckdb
import structlog

from src.research.models import ResearchReport

logger = structlog.get_logger("research_corpus")

RESEARCH_DB = "data/research_corpus.duckdb"


class ResearchCorpusStore:
    """Append-only DuckDB storage for ResearchReport objects."""

    def __init__(self, db_path: str = RESEARCH_DB):
        self._conn = duckdb.connect(db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS research_reports (
                report_id VARCHAR PRIMARY KEY,
                evaluation_version VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                sample_size INTEGER NOT NULL,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_reports_version
            ON research_reports (evaluation_version)
        """)

    def save(self, report: ResearchReport) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO research_reports
                (report_id, evaluation_version, status, sample_size, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                report.report_id,
                report.evaluation_version,
                report.status,
                report.sample_size,
                json.dumps(report.model_dump(mode="json")),
                report.generated_at.isoformat(),
            ],
        )
        logger.info(
            "Research report saved",
            report_id=report.report_id,
            status=report.status,
            sample_size=report.sample_size,
        )

    def list(self, limit: int = 10, offset: int = 0) -> list[ResearchReport]:
        rows = self._conn.execute(
            "SELECT payload_json FROM research_reports ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
        return [ResearchReport.model_validate(json.loads(r[0])) for r in rows]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
