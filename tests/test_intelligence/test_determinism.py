from __future__ import annotations

import json
from datetime import datetime

from src.intelligence.pipeline import ExperienceIntelligencePipeline
from src.intelligence.policies import AnalysisPolicy
from src.retrieval.models import RetrievalQuery, RetrievalRecord, SimilarityBreakdown
from src.retrieval.report import RankedResult, RetrievalReport


def _record(
    exp_id: str,
    pnl: float | None = 1.0,
    mae: float | None = 0.5,
    mfe: float | None = 2.0,
    bars: float | None = 10.0,
    integrity: int = 100,
    trend: str = "BULLISH",
) -> RetrievalRecord:
    return RetrievalRecord(
        experience_id=exp_id,
        position_id=f"pos-{exp_id}",
        created_at=datetime(2025, 1, 1),
        symbol="BTCUSDT",
        timeframe="5m",
        trend_regime=trend,
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        integrity_score=integrity,
        pnl_atr_multiple=pnl,
        mfe_atr_multiple=mfe,
        mae_atr_multiple=mae,
        bars_held=bars,
    )


def _make_report() -> RetrievalReport:
    records = [
        _record("exp-1", 2.0, 0.3, 3.0, 12),
        _record("exp-2", 1.5, 0.4, 2.5, 15),
        _record("exp-3", 0.8, 0.6, 1.8, 8),
        _record("exp-4", -0.2, 1.0, 0.5, 20),
        _record("exp-5", 3.0, 0.2, 4.0, 6),
        _record("exp-6", 0.5, 0.7, 1.2, 18),
        _record("exp-7", 1.2, 0.5, 2.2, 14),
        _record("exp-8", -0.8, 1.2, 0.3, 22),
        _record("exp-9", 0.0, 0.8, 1.0, 10),
        _record("exp-10", 2.5, 0.3, 3.5, 9),
    ]

    query = RetrievalQuery(symbol="BTCUSDT", timeframe="5m")
    results: list[RankedResult] = []
    for i, rec in enumerate(records):
        inv = len(records) - i
        score = SimilarityBreakdown(
            overall_score=inv / len(records),
            market_score=inv / len(records),
            execution_score=0.1,
            risk_score=0.2,
            outcome_score=0.1,
            context_score=0.1,
        )
        results.append(RankedResult(
            rank=i + 1,
            overall_similarity=score.overall_score,
            similarity_breakdown=score,
            record=rec,
        ))

    return RetrievalReport(
        query=query,
        results=results,
        filters_applied=["symbol=BTCUSDT", "timeframe=5m"],
        candidates_examined=10,
        candidates_returned=10,
    )


def test_experience_evidence_byte_identical():
    report = _make_report()
    pipeline = ExperienceIntelligencePipeline()
    policy = AnalysisPolicy(
        minimum_sample_size=5,
        minimum_integrity=50,
        pattern_threshold=0.3,
    )

    outputs = []
    for _ in range(10):
        evidence = pipeline.process(report, policy=policy)
        outputs.append(json.dumps(
            _exclude_timestamps(evidence),
            sort_keys=True,
        ))

    first = outputs[0]
    for i, o in enumerate(outputs[1:], 1):
        assert o == first, (
            f"Run {i + 1} differs — ExperienceEvidence is not deterministic"
        )


def test_prompt_context_byte_identical():
    report = _make_report()
    pipeline = ExperienceIntelligencePipeline()
    policy = AnalysisPolicy(
        minimum_sample_size=5,
        minimum_integrity=50,
        pattern_threshold=0.3,
    )

    evidence = pipeline.process(report, policy=policy)
    ctx1 = pipeline.generate_prompt_context(evidence)

    evidence2 = pipeline.process(report, policy=policy)
    ctx2 = pipeline.generate_prompt_context(evidence2)

    assert ctx1.model_dump() == ctx2.model_dump()


def test_insufficient_evidence_byte_identical():
    report = _make_report()
    pipeline = ExperienceIntelligencePipeline()
    policy = AnalysisPolicy(
        minimum_sample_size=1000,  # impossible to meet
        minimum_integrity=50,
    )

    outputs = []
    for _ in range(5):
        evidence = pipeline.process(report, policy=policy)
        outputs.append(json.dumps(
            _exclude_timestamps(evidence),
            sort_keys=True,
        ))

    first = outputs[0]
    for i, o in enumerate(outputs[1:], 1):
        assert o == first, (
            f"Insufficient evidence run {i + 1} differs"
        )


def _exclude_timestamps(data):
    """Remove provenance timestamps for comparison."""
    d = data.model_dump(mode="json")
    # provenance may contain version strings but no timestamps to remove
    return d
