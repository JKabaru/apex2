from __future__ import annotations

from datetime import datetime

from src.intelligence.pipeline import ExperienceIntelligencePipeline
from src.intelligence.policies import AnalysisPolicy
from src.retrieval.models import RetrievalQuery, RetrievalRecord, SimilarityBreakdown
from src.retrieval.report import RankedResult, RetrievalReport


def _report_with_records(records: list[RetrievalRecord]) -> RetrievalReport:
    results: list[RankedResult] = []
    for i, rec in enumerate(records):
        score = SimilarityBreakdown(overall_score=1.0)
        results.append(RankedResult(
            rank=i + 1,
            overall_similarity=1.0,
            similarity_breakdown=score,
            record=rec,
            experience_id=rec.experience_id,
            symbol=rec.symbol,
            timeframe=rec.timeframe,
            integrity_score=rec.integrity_score,
            created_at=rec.created_at,
        ))
    return RetrievalReport(
        query=RetrievalQuery(),
        results=results,
        candidates_examined=len(records),
        candidates_returned=len(records),
    )


def _record(
    pnl: float | None = 1.0,
    mae: float | None = 0.5,
    mfe: float | None = 2.0,
    bars: float | None = 10.0,
    integrity: int = 100,
) -> RetrievalRecord:
    return RetrievalRecord(
        experience_id="exp",
        position_id="pos",
        created_at=datetime(2025, 1, 1),
        symbol="BTCUSDT",
        timeframe="5m",
        trend_regime="BULLISH",
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        integrity_score=integrity,
        pnl_atr_multiple=pnl,
        mfe_atr_multiple=mfe,
        mae_atr_multiple=mae,
        bars_held=bars,
    )


def test_empty_report():
    report = _report_with_records([])
    pipeline = ExperienceIntelligencePipeline()
    evidence = pipeline.process(report)
    assert evidence.is_sufficient is False
    assert evidence.sample_size == 0


def test_all_low_integrity():
    records = [_record(integrity=30), _record(integrity=40)]
    report = _report_with_records(records)
    pipeline = ExperienceIntelligencePipeline()
    policy = AnalysisPolicy(minimum_integrity=80)
    evidence = pipeline.process(report, policy=policy)
    assert evidence.is_sufficient is False


def test_all_missing_pnl():
    records = [_record(pnl=None) for _ in range(10)]
    report = _report_with_records(records)
    pipeline = ExperienceIntelligencePipeline()
    policy = AnalysisPolicy(minimum_sample_size=5, minimum_integrity=80)
    evidence = pipeline.process(report, policy=policy)
    # PnL values list will be empty, validation should fail
    assert evidence.is_sufficient is False
    assert evidence.sample_size == 0


def test_single_outlier():
    records = [
        _record(pnl=0.5, mae=0.3, mfe=1.0, bars=5),
        _record(pnl=0.6, mae=0.4, mfe=1.2, bars=6),
        _record(pnl=0.4, mae=0.3, mfe=0.8, bars=7),
        _record(pnl=0.7, mae=0.5, mfe=1.5, bars=8),
        _record(pnl=0.5, mae=0.4, mfe=1.1, bars=6),
        _record(pnl=0.8, mae=0.3, mfe=1.3, bars=9),
        _record(pnl=0.6, mae=0.4, mfe=1.2, bars=7),
        _record(pnl=0.5, mae=0.3, mfe=1.0, bars=5),
        _record(pnl=0.7, mae=0.5, mfe=1.4, bars=8),
        _record(pnl=99.0, mae=0.3, mfe=1.0, bars=6),  # outlier
    ]
    report = _report_with_records(records)
    pipeline = ExperienceIntelligencePipeline()
    evidence = pipeline.process(report)
    assert evidence.outlier_count >= 1


def test_single_record():
    report = _report_with_records([_record()])
    pipeline = ExperienceIntelligencePipeline()
    evidence = pipeline.process(report)
    assert evidence.is_sufficient is False


def test_mixed_missing_metrics():
    records = [
        _record(pnl=1.0, mae=None, mfe=None, bars=None),
        _record(pnl=2.0, mae=0.5, mfe=None, bars=10.0),
        _record(pnl=None, mae=0.3, mfe=2.0, bars=8.0),
        _record(pnl=0.5, mae=0.4, mfe=1.0, bars=12.0),
        _record(pnl=0.8, mae=0.6, mfe=1.5, bars=15.0),
        _record(pnl=1.5, mae=0.3, mfe=2.5, bars=6.0),
    ]
    report = _report_with_records(records)
    pipeline = ExperienceIntelligencePipeline()
    policy = AnalysisPolicy(
        minimum_sample_size=3,
        minimum_integrity=80,
        pattern_threshold=0.5,
    )
    evidence = pipeline.process(report, policy=policy)
    # Should still compute partial results without crashing
    assert evidence.sample_size >= 3
    assert evidence.median_pnl_atr is not None


def test_large_values():
    records = [
        _record(pnl=1000.0, mae=50.0, mfe=2000.0, bars=1000.0),
        _record(pnl=2000.0, mae=75.0, mfe=3000.0, bars=2000.0),
        _record(pnl=1500.0, mae=60.0, mfe=2500.0, bars=1500.0),
        _record(pnl=1800.0, mae=80.0, mfe=2800.0, bars=1800.0),
        _record(pnl=2500.0, mae=90.0, mfe=3500.0, bars=2500.0),
        _record(pnl=1200.0, mae=55.0, mfe=2200.0, bars=1200.0),
    ]
    report = _report_with_records(records)
    pipeline = ExperienceIntelligencePipeline()
    evidence = pipeline.process(report)
    assert evidence.is_sufficient is True
    assert evidence.median_pnl_atr is not None
    assert evidence.median_mae_atr is not None
    assert evidence.median_mfe_atr is not None
    assert evidence.median_duration_bars is not None
