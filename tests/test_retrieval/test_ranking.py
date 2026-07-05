from __future__ import annotations

from datetime import datetime

from src.retrieval.models import RetrievalRecord, SimilarityBreakdown
from src.retrieval.ranking import RankingEngine


def _make_record(
    eid: str = "exp-1",
    integrity: int = 90,
    created: datetime = datetime(2025, 1, 1),
    **kw,
) -> RetrievalRecord:
    defaults = dict(
        position_id="pos-1",
        schema_version="2.0",
        pipeline_version="1.0",
        hash="abc",
        symbol="BTCUSDT",
        timeframe="5m",
        opportunity_id="opp-1",
        market_state_hash="",
        trend_regime="BULLISH",
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        normalized_entry_atr_multiple=None,
        normalized_exit_atr_multiple=None,
        pnl_atr_multiple=None,
        mfe_atr_multiple=None,
        mae_atr_multiple=None,
        entry_rsi_percentile=None,
        entry_volatility_percentile=None,
        holding_duration_minutes=None,
        bars_held=None,
        total_slippage_bps=None,
        total_fees_bps=None,
        realized_rr=None,
        initial_risk_atr_multiple=None,
    )
    defaults.update(kw)
    return RetrievalRecord(
        experience_id=eid,
        integrity_score=integrity,
        created_at=created,
        **defaults,
    )


def _make_score(overall: float) -> SimilarityBreakdown:
    return SimilarityBreakdown(
        market_score=0.5,
        execution_score=0.5,
        risk_score=0.5,
        outcome_score=0.5,
        context_score=0.5,
        overall_score=overall,
    )


def test_rank_by_overall_score_descending():
    engine = RankingEngine()
    records = [
        _make_record(eid="low"),
        _make_record(eid="high"),
    ]
    scores = [
        _make_score(0.3),
        _make_score(0.9),
    ]
    ranked = engine.rank(records, scores, max_results=10)
    assert ranked[0][0].experience_id == "high"
    assert ranked[1][0].experience_id == "low"


def test_rank_secondary_by_integrity():
    engine = RankingEngine()
    records = [
        _make_record(eid="low-int", integrity=50),
        _make_record(eid="high-int", integrity=99),
    ]
    scores = [
        _make_score(0.5),
        _make_score(0.5),
    ]
    ranked = engine.rank(records, scores, max_results=10)
    assert ranked[0][0].experience_id == "high-int"
    assert ranked[1][0].experience_id == "low-int"


def test_rank_tertiary_by_recency():
    engine = RankingEngine()
    records = [
        _make_record(eid="old", created=datetime(2024, 1, 1)),
        _make_record(eid="new", created=datetime(2025, 1, 1)),
    ]
    scores = [
        _make_score(0.5),
        _make_score(0.5),
    ]
    ranked = engine.rank(records, scores, max_results=10)
    assert ranked[0][0].experience_id == "new"
    assert ranked[1][0].experience_id == "old"


def test_rank_tiebreaker_by_experience_id():
    engine = RankingEngine()
    records = [
        _make_record(eid="b-aaaa", integrity=90, created=datetime(2025, 1, 1)),
        _make_record(eid="a-zzzz", integrity=90, created=datetime(2025, 1, 1)),
    ]
    scores = [
        _make_score(0.5),
        _make_score(0.5),
    ]
    ranked = engine.rank(records, scores, max_results=10)
    assert ranked[0][0].experience_id == "a-zzzz"
    assert ranked[1][0].experience_id == "b-aaaa"


def test_rank_deterministic():
    engine = RankingEngine()
    records = [
        _make_record(eid="c", integrity=80),
        _make_record(eid="a", integrity=95),
        _make_record(eid="b", integrity=90),
    ]
    scores = [
        _make_score(0.6),
        _make_score(0.9),
        _make_score(0.7),
    ]
    r1 = engine.rank(records, scores, max_results=10)
    r2 = engine.rank(records, scores, max_results=10)
    for (rec1, _), (rec2, _) in zip(r1, r2):
        assert rec1.experience_id == rec2.experience_id


def test_rank_respects_max_results():
    engine = RankingEngine()
    records = [_make_record(eid=f"exp-{i}") for i in range(10)]
    scores = [_make_score(0.5) for _ in range(10)]
    ranked = engine.rank(records, scores, max_results=3)
    assert len(ranked) == 3


def test_rank_raises_on_length_mismatch():
    engine = RankingEngine()
    records = [_make_record(eid="a")]
    scores = [_make_score(0.5), _make_score(0.5)]
    try:
        engine.rank(records, scores)
        assert False, "should have raised"
    except ValueError:
        pass
