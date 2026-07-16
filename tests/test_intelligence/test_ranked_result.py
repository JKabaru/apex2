from __future__ import annotations

from datetime import datetime

from src.retrieval.models import RetrievalRecord, SimilarityBreakdown
from src.retrieval.report import RankedResult


def _record() -> RetrievalRecord:
    return RetrievalRecord(
        experience_id="exp-1",
        position_id="pos-1",
        created_at=datetime(2025, 1, 1),
        symbol="BTCUSDT",
        timeframe="5m",
        trend_regime="BULLISH",
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        integrity_score=100,
        pnl_atr_multiple=1.0,
        mfe_atr_multiple=2.0,
        mae_atr_multiple=0.5,
        bars_held=10.0,
    )


def test_ranked_result_has_no_scalar_copies():
    rec = _record()
    result = RankedResult(
        rank=1,
        overall_similarity=0.95,
        similarity_breakdown=SimilarityBreakdown(),
        record=rec,
        experience_id=rec.experience_id,
        symbol=rec.symbol,
        timeframe=rec.timeframe,
        integrity_score=rec.integrity_score,
        created_at=rec.created_at,
    )
    # Must have these as direct fields (denormalized from record)
    assert result.experience_id == "exp-1"
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "5m"
    assert result.integrity_score == 100
    # Must access through record as well
    assert result.record.experience_id == "exp-1"
    assert result.record.symbol == "BTCUSDT"
    assert result.record.timeframe == "5m"
    assert result.record.integrity_score == 100


def test_ranked_result_retains_core_fields():
    rec = _record()
    result = RankedResult(
        rank=1,
        overall_similarity=0.95,
        similarity_breakdown=SimilarityBreakdown(),
        record=rec,
        experience_id=rec.experience_id,
        symbol=rec.symbol,
        timeframe=rec.timeframe,
        integrity_score=rec.integrity_score,
        created_at=rec.created_at,
    )
    assert result.rank == 1
    assert result.overall_similarity == 0.95
    assert isinstance(result.similarity_breakdown, SimilarityBreakdown)
    assert result.record is rec


def test_ranked_result_frozen():
    rec = _record()
    result = RankedResult(
        rank=1,
        overall_similarity=0.95,
        similarity_breakdown=SimilarityBreakdown(),
        record=rec,
        experience_id=rec.experience_id,
        symbol=rec.symbol,
        timeframe=rec.timeframe,
        integrity_score=rec.integrity_score,
        created_at=rec.created_at,
    )
    import pytest
    with pytest.raises(Exception):
        result.rank = 2
