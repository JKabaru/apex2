from __future__ import annotations

from src.intelligence.policies import AnalysisPolicy
from src.intelligence.validator import EvidenceValidator


def test_sufficient_when_all_criteria_met():
    policy = AnalysisPolicy(
        minimum_sample_size=5,
        minimum_integrity=80,
    )
    metrics = {
        "sample_size": 50,
        "avg_integrity": 95,
        "pnl_cv": 0.3,
    }
    result = EvidenceValidator.validate(metrics, policy)
    assert result.is_sufficient is True
    assert result.rejection_reasons == []


def test_insufficient_due_to_small_sample():
    policy = AnalysisPolicy(
        minimum_sample_size=20,
        minimum_integrity=80,
    )
    metrics = {
        "sample_size": 3,
        "avg_integrity": 95,
        "pnl_cv": 0.3,
    }
    result = EvidenceValidator.validate(metrics, policy)
    assert result.is_sufficient is False
    assert any("sample size" in r.lower() for r in result.rejection_reasons)


def test_insufficient_due_to_low_integrity():
    policy = AnalysisPolicy(
        minimum_sample_size=5,
        minimum_integrity=80,
    )
    metrics = {
        "sample_size": 50,
        "avg_integrity": 50,
        "pnl_cv": 0.3,
    }
    result = EvidenceValidator.validate(metrics, policy)
    assert result.is_sufficient is False
    assert any("integrity" in r.lower() for r in result.rejection_reasons)


def test_insufficient_due_to_high_variance():
    policy = AnalysisPolicy(
        minimum_sample_size=5,
        minimum_integrity=80,
    )
    metrics = {
        "sample_size": 50,
        "avg_integrity": 95,
        "pnl_cv": 2.5,
    }
    result = EvidenceValidator.validate(metrics, policy)
    assert result.is_sufficient is False
    assert any("cv" in r.lower() for r in result.rejection_reasons)


def test_insufficient_due_to_multiple_reasons():
    policy = AnalysisPolicy(
        minimum_sample_size=20,
        minimum_integrity=80,
    )
    metrics = {
        "sample_size": 3,
        "avg_integrity": 50,
        "pnl_cv": 2.5,
    }
    result = EvidenceValidator.validate(metrics, policy)
    assert result.is_sufficient is False
    assert len(result.rejection_reasons) >= 2


def test_empty_metrics_defaults():
    policy = AnalysisPolicy()
    metrics: dict = {}
    result = EvidenceValidator.validate(metrics, policy)
    assert result.is_sufficient is False
