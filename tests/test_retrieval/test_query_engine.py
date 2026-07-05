from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime

import duckdb
import pytest

from src.models.learning.trade_experience import (
    ConfigurationSnapshot,
    LearningExperience,
    LearningManifest,
    ManifestProvenance,
    NormalizedMetrics,
    OpportunityIdentity,
    ValidationReport,
)
from src.retrieval.models import RetrievalQuery
from src.retrieval.query_engine import QueryEngine
from src.storage.learning.learning_corpus import LearningCorpus


@pytest.fixture
def corpus_with_data():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.duckdb")
    corpus = LearningCorpus(db_path)
    corpus.create_schema()

    for i, (sym, tf, trend, vol, corr) in enumerate([
        ("BTCUSDT", "5m", "BULLISH", "HIGH", "STRONG"),
        ("ETHUSDT", "5m", "BEARISH", "LOW", "WEAK"),
        ("BTCUSDT", "15m", "BULLISH", "MEDIUM", "STRONG"),
        ("SOLUSDT", "5m", "NEUTRAL", "HIGH", "NEUTRAL"),
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
            highest_unrealized_profit=10.0,
            maximum_drawdown=2.0,
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
            verified_fields=["symbol", "timeframe"],
            missing_fields=[],
        )
        nm = NormalizedMetrics(
            normalized_entry_atr_multiple=1.5 + i * 0.1,
            pnl_atr_multiple=1.0 + i * 0.1,
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


def test_query_engine_returns_records(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery()
    results = engine.execute(query)
    assert len(results) == 4
    for r in results:
        assert r.experience_id.startswith("exp-")


def test_query_engine_filter_symbol(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery(symbol="BTCUSDT")
    results = engine.execute(query)
    assert all(r.symbol == "BTCUSDT" for r in results)
    assert len(results) == 2


def test_query_engine_filter_timeframe(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery(timeframe="15m")
    results = engine.execute(query)
    assert all(r.timeframe == "15m" for r in results)
    assert len(results) == 1


def test_query_engine_filter_regime(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery(trend_regime="BULLISH")
    results = engine.execute(query)
    assert all(r.trend_regime == "BULLISH" for r in results)
    assert len(results) == 2


def test_query_engine_filter_opportunity(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery(opportunity_id="opp-0")
    results = engine.execute(query)
    assert len(results) == 1
    assert results[0].opportunity_id == "opp-0"


def test_query_engine_filter_market_state_hash(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery(market_state_hash="hash-0")
    results = engine.execute(query)
    assert len(results) == 1
    assert results[0].market_state_hash == "hash-0"


def test_query_engine_deterministic(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery(symbol="BTCUSDT")
    r1 = engine.execute(query)
    r2 = engine.execute(query)
    ids1 = [r.experience_id for r in r1]
    ids2 = [r.experience_id for r in r2]
    assert ids1 == ids2


def test_query_engine_returns_retrieval_records(corpus_with_data):
    engine = QueryEngine(corpus_with_data)
    query = RetrievalQuery()
    results = engine.execute(query)
    for r in results:
        assert hasattr(r, "integrity_score")
        assert hasattr(r, "normalized_entry_atr_multiple")
        assert r.pipeline_version == "1.0"
        assert r.schema_version == "2.0"
