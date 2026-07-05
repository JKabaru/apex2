from __future__ import annotations

import math
from datetime import datetime

from src.retrieval.models import Explanation, RetrievalRecord, SimilarityBreakdown
from src.retrieval.similarity import (
    SimilarityEngine,
    _exact_match,
    _normalized_absolute_distance,
)
from src.retrieval.weights import SimilarityWeights
from src.learning.feature_catalog import _build_default_catalog


def _make_record(**overrides) -> RetrievalRecord:
    defaults = dict(
        experience_id="exp-1",
        position_id="pos-1",
        schema_version="2.0",
        pipeline_version="1.0",
        created_at=datetime(2025, 1, 1),
        hash="abc",
        symbol="BTCUSDT",
        timeframe="5m",
        opportunity_id="opp-1",
        market_state_hash="",
        trend_regime="BULLISH",
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        integrity_score=95,
        normalized_entry_atr_multiple=1.5,
        normalized_exit_atr_multiple=2.0,
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
    defaults.update(overrides)
    return RetrievalRecord(**defaults)


def test_normalized_absolute_distance_exact_match():
    assert _normalized_absolute_distance(1.5, 1.5, 5.0) == 1.0


def test_normalized_absolute_distance_far_apart():
    assert _normalized_absolute_distance(1.5, 6.5, 5.0) == 0.0


def test_normalized_absolute_distance_partial():
    result = _normalized_absolute_distance(1.0, 3.0, 5.0)
    expected = 1.0 - min(2.0 / 5.0, 1.0)
    assert math.isclose(result, expected)


def test_normalized_absolute_distance_both_none():
    assert _normalized_absolute_distance(None, None, 5.0) == 1.0


def test_normalized_absolute_distance_one_none():
    assert _normalized_absolute_distance(1.0, None, 5.0) == 0.5
    assert _normalized_absolute_distance(None, 1.0, 5.0) == 0.5


def test_exact_match_equal():
    assert _exact_match("BTCUSDT", "BTCUSDT") == 1.0


def test_exact_mismatch():
    assert _exact_match("BTCUSDT", "ETHUSDT") == 0.0


def test_exact_match_both_none():
    assert _exact_match(None, None) == 1.0


def test_exact_match_one_none():
    assert _exact_match("BTCUSDT", None) == 0.0
    assert _exact_match(None, "BTCUSDT") == 0.0


def test_similarity_identical_records():
    catalog = _build_default_catalog()
    engine = SimilarityEngine(catalog)
    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
        "market.trend_regime": "BULLISH",
        "market.normalized_entry_atr_multiple": 1.5,
        "market.normalized_exit_atr_multiple": 2.0,
        "market.entry_rsi_percentile": 0.7,
        "market.entry_volatility_percentile": 0.6,
        "execution.total_slippage_bps": 2.0,
        "execution.total_fees_bps": 4.0,
        "risk.initial_risk_atr_multiple": 2.0,
        "outcome.pnl_atr_multiple": 1.0,
        "outcome.mfe_atr_multiple": 2.5,
        "outcome.mae_atr_multiple": 0.5,
    }
    rec = _make_record()
    result = engine.compute_similarity(query_proj, rec)
    assert isinstance(result, SimilarityBreakdown)
    assert 0.0 <= result.overall_score <= 1.0


def test_similarity_context_exact_match():
    catalog = _build_default_catalog()
    engine = SimilarityEngine(catalog)
    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
        "market.correlation_regime": "STRONG",
    }
    rec = _make_record(symbol="BTCUSDT", timeframe="5m", correlation_regime="STRONG")
    result = engine.compute_similarity(query_proj, rec)
    assert result.context_score == 1.0


def test_similarity_context_mismatch():
    catalog = _build_default_catalog()
    engine = SimilarityEngine(catalog)
    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
    }
    rec = _make_record(symbol="ETHUSDT", timeframe="15m")
    result = engine.compute_similarity(query_proj, rec)
    assert result.context_score == 0.0


def test_similarity_deterministic():
    catalog = _build_default_catalog()
    engine = SimilarityEngine(catalog)
    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
        "market.trend_regime": "BULLISH",
    }
    rec = _make_record()
    r1 = engine.compute_similarity(query_proj, rec)
    r2 = engine.compute_similarity(query_proj, rec)
    assert r1.overall_score == r2.overall_score
    assert r1.market_score == r2.market_score
    assert r1.context_score == r2.context_score


def test_similarity_breakdown_contains_details():
    catalog = _build_default_catalog()
    engine = SimilarityEngine(catalog)
    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
    }
    rec = _make_record()
    result = engine.compute_similarity(query_proj, rec)
    assert "market" in result.group_details
    assert "context" in result.group_details
    assert len(result.explanations) > 0


def test_similarity_with_none_values():
    catalog = _build_default_catalog()
    engine = SimilarityEngine(catalog)
    query_proj: dict = {}
    rec = _make_record(
        normalized_entry_atr_multiple=None,
        total_slippage_bps=None,
    )
    result = engine.compute_similarity(query_proj, rec)
    assert 0.0 <= result.overall_score <= 1.0


def test_similarity_custom_weights():
    catalog = _build_default_catalog()
    engine = SimilarityEngine(catalog, weights=SimilarityWeights(
        market_weight=1.0, execution_weight=0.0, risk_weight=0.0,
        context_weight=0.0, outcome_weight=0.0,
    ))
    query_proj = {
        "context.symbol": "BTCUSDT",
        "context.timeframe": "5m",
        "market.trend_regime": "BULLISH",
        "market.normalized_entry_atr_multiple": 1.5,
    }
    rec = _make_record(symbol="BTCUSDT", timeframe="5m", trend_regime="BULLISH",
                       normalized_entry_atr_multiple=1.5)
    result = engine.compute_similarity(query_proj, rec)
    assert result.overall_score == result.market_score
