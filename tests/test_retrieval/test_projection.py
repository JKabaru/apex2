from __future__ import annotations

from datetime import datetime

from src.models.learning.trade_experience import (
    LearningExperience,
    LearningManifest,
    NormalizedMetrics,
    OpportunityIdentity,
    ValidationReport,
)
from src.retrieval.models import RetrievalQuery
from src.retrieval.projection import CorpusProjection


def _make_test_manifest(**overrides) -> LearningManifest:
    exp = LearningExperience(
        experience_id="exp-1",
        position_id="pos-1",
        symbol="BTCUSDT",
        timeframe="5m",
        entry_price=100.0,
        exit_price=105.0,
        fees=0.1,
        exit_fees=0.1,
        highest_unrealized_profit=10.0,
        maximum_drawdown=2.0,
        slippage_bps=1.0,
        spread_bps=1.0,
        entry_atr=2.0,
        entry_rsi=60.0,
        exit_atr=None,
        exit_rsi=None,
        trend_regime="BULLISH",
        volatility_regime="HIGH",
        correlation_regime="STRONG",
        opportunity_id="opp-1",
    )
    val = ValidationReport(
        verified_fields=["symbol", "timeframe", "entry_price"],
    )
    nm = NormalizedMetrics(
        normalized_entry_atr_multiple=1.5,
        pnl_atr_multiple=1.0,
        mfe_atr_multiple=2.5,
        mae_atr_multiple=0.5,
        entry_rsi_percentile=0.7,
    )
    opp = OpportunityIdentity(
        opportunity_id="opp-1",
        market_state_hash="abc123",
        discovered_at=datetime(2025, 1, 1),
        anchor_symbol="BTCUSDT",
        symbol="BTCUSDT",
        timeframe="5m",
    )
    manifest = LearningManifest(
        experience_id="exp-1",
        position_id="pos-1",
        learning_experience=exp,
        validation_report=val,
        normalized_metrics=nm,
        opportunity_identity=opp,
    )
    return manifest


def test_project_manifest_produces_retrieval_record():
    projector = CorpusProjection()
    manifest = _make_test_manifest()
    record = projector.project_manifest(manifest)

    assert record.experience_id == "exp-1"
    assert record.symbol == "BTCUSDT"
    assert record.timeframe == "5m"
    assert record.trend_regime == "BULLISH"
    assert record.volatility_regime == "HIGH"
    assert record.correlation_regime == "STRONG"
    assert record.integrity_score == 100
    assert record.normalized_entry_atr_multiple == 1.5
    assert record.pnl_atr_multiple == 1.0
    assert record.mfe_atr_multiple == 2.5
    assert record.mae_atr_multiple == 0.5
    assert record.entry_rsi_percentile == 0.7
    assert record.opportunity_id == "opp-1"
    assert record.market_state_hash == "abc123"


def test_project_manifest_integrity_score():
    projector = CorpusProjection()
    exp = LearningExperience(
        experience_id="exp-2", position_id="pos-2",
        symbol="ETHUSDT", timeframe="5m",
        entry_price=100.0, exit_price=None,
        fees=0.0, exit_fees=None,
        highest_unrealized_profit=0.0, maximum_drawdown=0.0,
        slippage_bps=None, spread_bps=None,
        entry_atr=None, entry_rsi=None,
        exit_atr=None, exit_rsi=None,
        trend_regime=None, volatility_regime=None, correlation_regime=None,
    )
    val = ValidationReport(
        missing_fields=["exit_price", "exit_fees", "entry_atr", "entry_rsi", "trend_regime"],
    )
    nm = NormalizedMetrics()
    manifest = LearningManifest(
        experience_id="exp-2", position_id="pos-2",
        learning_experience=exp, validation_report=val, normalized_metrics=nm,
    )
    record = projector.project_manifest(manifest)
    expected_integrity = 100 - 5 * 5
    assert record.integrity_score == expected_integrity


def test_project_query():
    projector = CorpusProjection()
    query = RetrievalQuery(
        symbol="BTCUSDT",
        timeframe="5m",
        trend_regime="BULLISH",
    )
    proj = projector.project_query(query)
    assert proj["symbol"] == "BTCUSDT"
    assert proj["timeframe"] == "5m"
    assert proj["trend_regime"] == "BULLISH"
    assert "opportunity_id" not in proj
