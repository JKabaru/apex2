from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence

from src.evaluation.models import DecisionEvaluation
from src.research.statistics import compute_wilson_interval


def compute_effect_size(
    mean_test: float,
    std_test: float,
    mean_control: float,
    std_control: float,
) -> float:
    if std_test <= 0 and std_control <= 0:
        return 0.0
    pooled = math.sqrt((std_test**2 + std_control**2) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (mean_test - mean_control) / pooled


def compute_information_weight(evaluation: DecisionEvaluation) -> float:
    score = 1.0

    # Surprise factor: |confidence - outcome|
    if evaluation.was_profitable is not None:
        outcome_val = 1.0 if evaluation.was_profitable else 0.0
        surprise = abs(evaluation.llm_confidence - outcome_val)
        score += surprise * 0.5

    # Regime rarity: COLD_START carries more signal
    if evaluation.evidence_source == "COLD_START":
        score += 0.3

    # Drawdown extremity: high-stress trades carry information
    if evaluation.actual_max_drawdown > 0.1:
        score += 0.2

    # Integrity penalty: low-integrity data is less valuable
    if evaluation.actual_integrity_score < 50:
        score -= 0.3

    return max(0.5, min(2.0, score))


def compute_evidence_strength(
    test_wins: int,
    test_total: int,
    control_wins: int,
    control_total: int,
) -> tuple[float, str, float, float]:
    if test_total < 2 or control_total < 2:
        return (0.0, "NEGLIGIBLE", 0.0, 0.0)

    p_test = test_wins / test_total
    p_control = control_wins / control_total

    std_test = math.sqrt(p_test * (1 - p_test))
    std_control = math.sqrt(p_control * (1 - p_control))

    d = compute_effect_size(p_test, std_test, p_control, std_control)

    if abs(d) >= 0.8:
        label = "LARGE"
    elif abs(d) >= 0.5:
        label = "MEDIUM"
    elif abs(d) >= 0.2:
        label = "SMALL"
    else:
        label = "NEGLIGIBLE"

    ci_low, ci_high = compute_wilson_interval(test_wins, test_total)
    return (d, label, ci_low, ci_high)


def compute_evidence_quality(
    evaluations: Sequence[DecisionEvaluation],
    supporting_ids: set[str],
    conflicting_ids: set[str],
) -> tuple[float, int, float, float, str]:
    total = len(evaluations)
    if total == 0:
        return (0.0, 0, 0.0, 0.0, "LOW")

    total_info_weight = sum(
        compute_information_weight(e) for e in evaluations
    )

    evidence_sources: set[str] = set()
    for e in evaluations:
        evidence_sources.add(e.evidence_source)
    regime_count = len(evidence_sources)
    cross_regime = min(1.0, regime_count / 4.0)

    win_rates: list[float] = []
    for source in evidence_sources:
        group = [e for e in evaluations if e.evidence_source == source]
        if len(group) >= 3:
            wr = sum(1 for e in group if e.was_profitable) / len(group)
            win_rates.append(wr)
    consistency = 1.0 - (statistics.stdev(win_rates) if len(win_rates) >= 2 else 0.0)
    consistency = max(0.0, min(1.0, consistency))

    conflict_ratio = len(conflicting_ids) / max(1, len(supporting_ids) + len(conflicting_ids))
    if conflict_ratio < 0.1:
        trust = "HIGH"
    elif conflict_ratio < 0.3:
        trust = "MEDIUM"
    else:
        trust = "LOW"

    return (round(total_info_weight, 4), total, round(cross_regime, 4), round(consistency, 4), trust)


def determine_confidence_tier(
    evidence_strength: tuple[float, str, float, float],
    evidence_quality: tuple[float, int, float, float, str],
) -> str:
    _, mag_label, _, _ = evidence_strength
    _, _, _, _, trust = evidence_quality

    if mag_label == "LARGE" and trust == "HIGH":
        return "HIGH"
    elif mag_label in ("LARGE", "MEDIUM") and trust in ("HIGH", "MEDIUM"):
        return "MEDIUM"
    else:
        return "LOW"
