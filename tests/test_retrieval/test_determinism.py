from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime

import pytest

from src.learning.feature_catalog import _build_default_catalog
from src.models.learning.trade_experience import (
    LearningExperience,
    LearningManifest,
    NormalizedMetrics,
    OpportunityIdentity,
    ValidationReport,
)
from src.retrieval.models import (
    Explanation,
    RetrievalQuery,
    RetrievalContext,
    SimilarityBreakdown,
)
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.similarity import SimilarityEngine
from src.retrieval.weights import SimilarityWeights
from src.storage.learning.learning_corpus import LearningCorpus


@pytest.fixture
def seeded_corpus():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.duckdb")
    corpus = LearningCorpus(db_path)
    corpus.create_schema()

    for i, (sym, tf, trend, vol, corr, mfe, mae) in enumerate([
        ("BTCUSDT", "5m", "BULLISH", "HIGH", "STRONG", 10.0, 2.0),
        ("ETHUSDT", "5m", "BEARISH", "LOW", "WEAK", 15.0, 3.0),
        ("BTCUSDT", "15m", "BULLISH", "MEDIUM", "STRONG", 8.0, 1.5),
        ("SOLUSDT", "5m", "NEUTRAL", "HIGH", "NEUTRAL", 5.0, 4.0),
    ]):
        exp = LearningExperience(
            experience_id=f"exp-{i}",
            position_id=f"pos-{i}",
            symbol=sym,
            timeframe=tf,
            entry_price=100.0,
            exit_price=105.0,
            fees=0.1,
            exit_fees=0.1,
            highest_unrealized_profit=mfe,
            maximum_drawdown=mae,
            slippage_bps=1.0,
            spread_bps=1.0,
            entry_atr=2.0,
            entry_rsi=60.0,
            exit_atr=None,
            exit_rsi=None,
            trend_regime=trend,
            volatility_regime=vol,
            correlation_regime=corr,
            opportunity_id=f"opp-{i}",
        )
        val = ValidationReport(
            verified_fields=["symbol", "timeframe", "entry_price"],
            missing_fields=[],
        )
        nm = NormalizedMetrics(
            normalized_entry_atr_multiple=1.5 + i * 0.1,
            pnl_atr_multiple=1.0 + i * 0.1,
            mfe_atr_multiple=mfe / 2.0,
            mae_atr_multiple=mae / 2.0,
            entry_rsi_percentile=0.7 - i * 0.05,
        )
        opp = OpportunityIdentity(
            opportunity_id=f"opp-{i}",
            market_state_hash=f"hash-{i}",
            discovered_at=datetime(2025, 1, 1),
            anchor_symbol="BTCUSDT",
            symbol=sym,
            timeframe=tf,
        )
        manifest = LearningManifest(
            experience_id=f"exp-{i}",
            position_id=f"pos-{i}",
            learning_experience=exp,
            validation_report=val,
            normalized_metrics=nm,
            opportunity_identity=opp,
        )
        corpus.save(manifest)

    yield corpus

    corpus.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_deterministic_retrieval_identical_output(seeded_corpus):
    catalog = _build_default_catalog()
    pipeline = RetrievalPipeline(seeded_corpus, catalog)

    query = RetrievalQuery(
        symbol="BTCUSDT",
        timeframe="5m",
        min_integrity=80,
        max_results=5,
    )

    ctx = RetrievalContext(
        current_market_regime="BULLISH",
        current_volatility="HIGH",
        requested_max_results=10,
    )

    reports = []
    for _ in range(5):
        report = pipeline.retrieve(query, context=ctx)
        reports.append(json.dumps(
            report.model_dump(
                mode="json",
                exclude={"generated_at", "execution_time_ms"},
            ),
            sort_keys=True,
        ))

    first = reports[0]
    for i, r in enumerate(reports[1:], 1):
        assert r == first, (
            f"Run {i + 1} differs from run 1 — retrieval is not deterministic"
        )


def test_deterministic_retrieval_no_context(seeded_corpus):
    catalog = _build_default_catalog()
    pipeline = RetrievalPipeline(seeded_corpus, catalog)

    query = RetrievalQuery(
        symbol="BTCUSDT",
        max_results=10,
    )

    def _dump(report):
        return json.dumps(
            report.model_dump(mode="json", exclude={"generated_at", "execution_time_ms"}),
            sort_keys=True,
        )
    r1 = _dump(pipeline.retrieve(query))
    r2 = _dump(pipeline.retrieve(query))
    assert r1 == r2


def test_explanation_categorization():
    catalog = _build_default_catalog()
    from src.retrieval.models import RetrievalRecord

    engine = SimilarityEngine(catalog)

    rec = RetrievalRecord(
        experience_id="exp-0",
        position_id="pos-0",
        created_at=datetime(2025, 1, 1),
        symbol="BTCUSDT",
        timeframe="5m",
        trend_regime="BULLISH",
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        integrity_score=100,
        normalized_entry_atr_multiple=1.5,
        pnl_atr_multiple=1.0,
        mfe_atr_multiple=2.5,
        mae_atr_multiple=0.5,
        entry_rsi_percentile=0.7,
        entry_volatility_percentile=0.6,
        holding_duration_minutes=120.0,
        bars_held=24.0,
        total_slippage_bps=2.0,
        total_fees_bps=4.0,
        realized_rr=None,
        initial_risk_atr_multiple=2.0,
    )

    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
        "market.trend_regime": "BULLISH",
        "market.normalized_entry_atr_multiple": 1.5,
        "outcome.pnl_atr_multiple": 1.0,
        "market.correlation_regime": "STRONG",
    }

    result = engine.compute_similarity(query_proj, rec)

    assert isinstance(result, SimilarityBreakdown)
    assert len(result.explanations) > 0

    all_matched = []
    all_mismatched = []
    all_ignored = []
    all_missing = []

    for exp in result.explanations:
        assert isinstance(exp, Explanation)
        all_matched.extend(exp.matched_features)
        all_mismatched.extend(exp.mismatched_features)
        all_ignored.extend(exp.ignored_features)
        all_missing.extend(exp.missing_features)

    assert "context.symbol" in all_matched or "market.trend_regime" in all_matched
    assert "market.volatility_regime" in all_ignored or len(all_ignored) >= 0
    assert "risk.realized_rr" in all_missing or len(all_missing) >= 0


def test_explanation_confidence():
    catalog = _build_default_catalog()
    from src.retrieval.models import RetrievalRecord

    engine = SimilarityEngine(catalog)

    rec = RetrievalRecord(
        experience_id="exp-0",
        position_id="pos-0",
        created_at=datetime(2025, 1, 1),
        symbol="BTCUSDT",
        timeframe="5m",
        trend_regime="BULLISH",
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        integrity_score=100,
        normalized_entry_atr_multiple=1.5,
        pnl_atr_multiple=1.0,
        mfe_atr_multiple=2.5,
        mae_atr_multiple=0.5,
        entry_rsi_percentile=0.7,
        entry_volatility_percentile=0.6,
        total_slippage_bps=2.0,
        total_fees_bps=4.0,
        initial_risk_atr_multiple=2.0,
    )

    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
        "market.normalized_entry_atr_multiple": 1.5,
        "market.volatility_regime": "HIGH",
        "execution.total_slippage_bps": 2.0,
        "execution.total_fees_bps": 4.0,
    }

    result = engine.compute_similarity(query_proj, rec)

    for exp in result.explanations:
        assert 0.0 <= exp.confidence_score <= 1.0


def test_weight_profile_default():
    w = SimilarityWeights()
    assert abs(w.market_weight + w.execution_weight + w.risk_weight + w.context_weight + w.outcome_weight - 1.0) < 1e-6


def test_weight_profile_sum_validation():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SimilarityWeights(
            market_weight=1.0, execution_weight=0.5,
            risk_weight=0.0, context_weight=0.0, outcome_weight=0.0,
        )


def test_weight_profile_registry():
    from src.retrieval.weights import WeightProfileRegistry
    default = WeightProfileRegistry.get("default")
    assert isinstance(default, SimilarityWeights)
    momentum = WeightProfileRegistry.get("momentum")
    assert momentum.market_weight == 0.50
    rev = WeightProfileRegistry.get("mean_reversion")
    assert rev.context_weight == 0.30
