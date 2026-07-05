from __future__ import annotations

from datetime import datetime

from src.intelligence.patterns import PatternExtractor
from src.intelligence.policies import AnalysisPolicy
from src.retrieval.models import RetrievalRecord


def _record(pnl: float | None, trend: str) -> RetrievalRecord:
    return RetrievalRecord(
        experience_id="exp",
        position_id="pos",
        created_at=datetime(2025, 1, 1),
        symbol="BTCUSDT",
        timeframe="5m",
        trend_regime=trend,
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        integrity_score=100,
        pnl_atr_multiple=pnl,
        mfe_atr_multiple=1.0,
        mae_atr_multiple=0.5,
        bars_held=10.0,
    )


def test_extract_success_patterns():
    policy = AnalysisPolicy(minimum_sample_size=3, pattern_threshold=0.5)
    records = [
        _record(1.0, "BULLISH"),
        _record(2.0, "BULLISH"),
        _record(0.5, "BEARISH"),
        _record(-1.0, "BULLISH"),
        _record(-0.5, "BEARISH"),
    ]
    patterns = PatternExtractor.extract_patterns(
        records, "trend_regime",
        lambda r: r.pnl_atr_multiple is not None and r.pnl_atr_multiple > 0,
        policy,
    )
    assert len(patterns) > 0
    bullish_pat = [p for p in patterns if p.value == "BULLISH"]
    assert len(bullish_pat) == 1
    assert bullish_pat[0].frequency >= 0.5
    assert 0.0 <= bullish_pat[0].confidence_score <= 1.0


def test_extract_failure_patterns():
    policy = AnalysisPolicy(minimum_sample_size=2, pattern_threshold=0.4)
    records = [
        _record(1.0, "BULLISH"),
        _record(-1.0, "BEARISH"),
        _record(-2.0, "BEARISH"),
        _record(-0.5, "NEUTRAL"),
    ]
    patterns = PatternExtractor.extract_patterns(
        records, "trend_regime",
        lambda r: r.pnl_atr_multiple is not None and r.pnl_atr_multiple <= 0,
        policy,
    )
    assert len(patterns) >= 1
    bearish_pat = [p for p in patterns if p.value == "BEARISH"]
    assert len(bearish_pat) >= 1
    assert bearish_pat[0].frequency > 0.0


def test_insufficient_sample_returns_empty():
    policy = AnalysisPolicy(minimum_sample_size=100, pattern_threshold=0.5)
    records = [_record(1.0, "BULLISH"), _record(2.0, "BULLISH")]
    patterns = PatternExtractor.extract_patterns(
        records, "trend_regime",
        lambda r: r.pnl_atr_multiple is not None and r.pnl_atr_multiple > 0,
        policy,
    )
    assert patterns == []


def test_empty_records_returns_empty():
    policy = AnalysisPolicy()
    patterns = PatternExtractor.extract_patterns(
        [], "trend_regime",
        lambda r: r.pnl_atr_multiple is not None and r.pnl_atr_multiple > 0,
        policy,
    )
    assert patterns == []


def test_generic_predicate_works_with_any_field():
    policy = AnalysisPolicy(minimum_sample_size=3, pattern_threshold=0.5)
    records = [
        _record(1.0, "BULLISH"),
        _record(2.0, "BULLISH"),
        _record(0.5, "BEARISH"),
        _record(-1.0, "BULLISH"),
    ]
    volatility_patterns = PatternExtractor.extract_patterns(
        records, "volatility_regime",
        lambda r: r.pnl_atr_multiple is not None and r.pnl_atr_multiple > 0,
        policy,
    )
    for p in volatility_patterns:
        assert p.field == "volatility_regime"
        assert p.value in ("HIGH", "LOW", "MEDIUM")
