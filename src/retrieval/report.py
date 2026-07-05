from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.retrieval.models import RetrievalQuery, RetrievalRecord, RetrievalContext, SimilarityBreakdown


class RankedResult(BaseModel, frozen=True):
    rank: int
    experience_id: str
    symbol: str
    timeframe: str
    overall_similarity: float
    similarity_breakdown: SimilarityBreakdown
    integrity_score: int
    created_at: datetime
    record: RetrievalRecord


class RetrievalReport(BaseModel, frozen=True):
    query: RetrievalQuery
    context: Optional[RetrievalContext] = None
    filters_applied: list[str] = Field(default_factory=list)
    candidates_examined: int = 0
    candidates_rejected: int = 0
    candidates_returned: int = 0
    results: list[RankedResult] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    pipeline_version: str = "4.4.1"
    catalog_version: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


def build_report(
    query: RetrievalQuery,
    candidates_before_filter: int,
    candidates_after_filter: int,
    ranked: list[tuple[Any, SimilarityBreakdown]],
    execution_time_ms: float,
    catalog_version: str = "",
    context: Optional[RetrievalContext] = None,
) -> RetrievalReport:
    filters_applied: list[str] = []
    if query.symbol is not None:
        filters_applied.append(f"symbol={query.symbol}")
    if query.timeframe is not None:
        filters_applied.append(f"timeframe={query.timeframe}")
    if query.opportunity_id is not None:
        filters_applied.append(f"opportunity_id={query.opportunity_id}")
    if query.market_state_hash is not None:
        filters_applied.append(f"market_state_hash={query.market_state_hash}")
    if query.trend_regime is not None:
        filters_applied.append(f"trend_regime={query.trend_regime}")
    if query.volatility_regime is not None:
        filters_applied.append(f"volatility_regime={query.volatility_regime}")
    if query.correlation_regime is not None:
        filters_applied.append(f"correlation_regime={query.correlation_regime}")
    if query.min_integrity > 0:
        filters_applied.append(f"min_integrity>={query.min_integrity}")

    results: list[RankedResult] = []
    for rank_idx, (rec, score) in enumerate(ranked):
        results.append(RankedResult(
            rank=rank_idx + 1,
            experience_id=rec.experience_id,
            symbol=rec.symbol,
            timeframe=rec.timeframe,
            overall_similarity=score.overall_score,
            similarity_breakdown=score,
            integrity_score=rec.integrity_score,
            created_at=rec.created_at,
            record=rec,
        ))

    return RetrievalReport(
        query=query,
        context=context,
        filters_applied=filters_applied,
        candidates_examined=candidates_before_filter,
        candidates_rejected=candidates_before_filter - candidates_after_filter,
        candidates_returned=len(results),
        results=results,
        execution_time_ms=round(execution_time_ms, 3),
        pipeline_version="4.4.1",
        catalog_version=catalog_version,
    )
