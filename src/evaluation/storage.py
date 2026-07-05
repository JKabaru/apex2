from __future__ import annotations

import json
from typing import Optional

import duckdb
import structlog

from src.evaluation.models import DecisionEvaluation

logger = structlog.get_logger("evaluation_corpus")

EVALUATION_DB = "data/evaluation_corpus.duckdb"


class EvaluationCorpus:
    """Append-only DuckDB storage for DecisionEvaluation objects."""

    def __init__(self, db_path: str = EVALUATION_DB):
        self._conn = duckdb.connect(db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id VARCHAR PRIMARY KEY,
                position_id VARCHAR NOT NULL,
                opportunity_id VARCHAR NOT NULL,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_position_id
            ON evaluations (position_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_opportunity_id
            ON evaluations (opportunity_id)
        """)

    def save(self, evaluation: DecisionEvaluation) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO evaluations
                (evaluation_id, position_id, opportunity_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                evaluation.evaluation_id,
                evaluation.position_id,
                evaluation.opportunity_id,
                json.dumps(evaluation.model_dump(mode="json")),
                evaluation.created_at.isoformat(),
            ],
        )
        logger.info(
            "Evaluation saved",
            evaluation_id=evaluation.evaluation_id,
            position_id=evaluation.position_id,
            confidence_vs_outcome=evaluation.confidence_vs_outcome,
            was_profitable=evaluation.was_profitable,
        )

    def get_by_position_id(self, position_id: str) -> Optional[DecisionEvaluation]:
        row = self._conn.execute(
            "SELECT payload_json FROM evaluations WHERE position_id = ?",
            [position_id],
        ).fetchone()
        if row is None:
            return None
        return DecisionEvaluation.model_validate(json.loads(row[0]))

    def get_by_opportunity_id(self, opportunity_id: str) -> Optional[DecisionEvaluation]:
        row = self._conn.execute(
            "SELECT payload_json FROM evaluations WHERE opportunity_id = ?",
            [opportunity_id],
        ).fetchone()
        if row is None:
            return None
        return DecisionEvaluation.model_validate(json.loads(row[0]))

    def list(self, limit: int = 50, offset: int = 0) -> list[DecisionEvaluation]:
        if limit == 0:
            rows = self._conn.execute(
                "SELECT payload_json FROM evaluations ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM evaluations ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [limit, offset],
            ).fetchall()
        return [DecisionEvaluation.model_validate(json.loads(r[0])) for r in rows]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
