from __future__ import annotations

import statistics
from typing import Sequence

from src.evaluation.models import DecisionEvaluation
from src.research.models import BiasFinding


class PatternDiscoveryEngine:
    """Deterministic bias detection over a list of DecisionEvaluation objects.
    All thresholds are configurable via kwargs with sane defaults."""

    @staticmethod
    def discover_biases(
        evaluations: Sequence[DecisionEvaluation],
        **kwargs,
    ) -> list[BiasFinding]:
        # Unpack all thresholds from kwargs with defaults
        min_evals = kwargs.get("min_evals", 10)
        high_conf_threshold = kwargs.get("high_conf_threshold", 0.5)
        min_high_conf_evals = kwargs.get("min_high_conf_evals", 10)
        overconfidence_high_gap = kwargs.get("overconfidence_high_gap", 0.30)
        overconfidence_medium_gap = kwargs.get("overconfidence_medium_gap", 0.15)
        min_low_conf_evals = kwargs.get("min_low_conf_evals", 10)
        underconfidence_wr = kwargs.get("underconfidence_wr", 0.60)
        min_side_evals = kwargs.get("min_side_evals", 5)
        side_imbalance = kwargs.get("side_imbalance", 0.20)
        min_tier_evals = kwargs.get("min_tier_evals", 5)
        stop_loss_rate = kwargs.get("stop_loss_rate", 0.40)
        min_duration_evals = kwargs.get("min_duration_evals", 5)
        short_hold_wr = kwargs.get("short_hold_wr", 0.40)

        findings: list[BiasFinding] = []

        valid = [e for e in evaluations if e.was_profitable is not None]
        if len(valid) < min_evals:
            return findings

        # --- OVERCONFIDENCE ---
        high_conf = [e for e in valid if e.llm_confidence >= high_conf_threshold]
        if len(high_conf) >= min_high_conf_evals:
            high_conf_wins = sum(1 for e in high_conf if e.was_profitable)
            high_conf_wr = high_conf_wins / len(high_conf)
            avg_conf = statistics.mean(e.llm_confidence for e in high_conf)
            gap = avg_conf - high_conf_wr
            if gap > overconfidence_high_gap:
                findings.append(
                    BiasFinding(
                        bias_type="OVERCONFIDENCE",
                        severity="HIGH",
                        description=(
                            f"High-confidence trades (n={len(high_conf)}) have win rate "
                            f"{high_conf_wr:.1%} vs avg confidence {avg_conf:.1%} (gap={gap:.1%})"
                        ),
                        metric_value=round(gap, 4),
                        threshold=overconfidence_high_gap,
                    )
                )
            elif gap > overconfidence_medium_gap:
                findings.append(
                    BiasFinding(
                        bias_type="OVERCONFIDENCE",
                        severity="MEDIUM",
                        description=(
                            f"High-confidence trades (n={len(high_conf)}) have win rate "
                            f"{high_conf_wr:.1%} vs avg confidence {avg_conf:.1%} (gap={gap:.1%})"
                        ),
                        metric_value=round(gap, 4),
                        threshold=overconfidence_medium_gap,
                    )
                )

        # --- UNDERCONFIDENCE ---
        low_conf = [e for e in valid if e.llm_confidence < high_conf_threshold]
        if len(low_conf) >= min_low_conf_evals:
            low_conf_wins = sum(1 for e in low_conf if e.was_profitable)
            low_conf_wr = low_conf_wins / len(low_conf)
            if low_conf_wr > underconfidence_wr:
                findings.append(
                    BiasFinding(
                        bias_type="UNDERCONFIDENCE",
                        severity="MEDIUM",
                        description=(
                            f"Low-confidence trades (n={len(low_conf)}) have win rate "
                            f"{low_conf_wr:.1%}, suggesting the model undervalues its own predictions"
                        ),
                        metric_value=round(low_conf_wr, 4),
                        threshold=underconfidence_wr,
                    )
                )

        # --- LONG_BIAS ---
        buy_trades = [e for e in valid if e.llm_action == "BUY"]
        sell_trades = [e for e in valid if e.llm_action == "SELL"]
        if len(buy_trades) >= min_side_evals and len(sell_trades) >= min_side_evals:
            buy_wr = sum(1 for e in buy_trades if e.was_profitable) / len(buy_trades)
            sell_wr = (
                sum(1 for e in sell_trades if e.was_profitable) / len(sell_trades)
            )
            diff = abs(buy_wr - sell_wr)
            if diff > side_imbalance:
                dominant = "BUY" if buy_wr > sell_wr else "SELL"
                findings.append(
                    BiasFinding(
                        bias_type="LONG_BIAS",
                        severity="HIGH",
                        description=(
                            f"{dominant} trades outperform the opposite side by {diff:.1%} "
                            f"(BUY: {buy_wr:.1%}, SELL: {sell_wr:.1%})"
                        ),
                        metric_value=round(diff, 4),
                        threshold=side_imbalance,
                    )
                )

        # --- EVIDENCE_TIER_INEFFECTIVENESS ---
        exact = [e for e in valid if e.evidence_source == "EXACT"]
        cold = [e for e in valid if e.evidence_source == "COLD_START"]
        if len(exact) >= min_tier_evals and len(cold) >= min_tier_evals:
            exact_wr = sum(1 for e in exact if e.was_profitable) / len(exact)
            cold_wr = sum(1 for e in cold if e.was_profitable) / len(cold)
            if cold_wr >= exact_wr:
                findings.append(
                    BiasFinding(
                        bias_type="EVIDENCE_TIER_INEFFECTIVENESS",
                        severity="MEDIUM",
                        description=(
                            f"COLD_START evidence (n={len(cold)}) performs as well or better than "
                            f"EXACT evidence (n={len(exact)}) — EXACT: {exact_wr:.1%}, COLD_START: {cold_wr:.1%}"
                        ),
                        metric_value=round(exact_wr - cold_wr, 4),
                        threshold=0.0,
                    )
                )

        # --- STOP_LOSS_FREQUENCY ---
        stop_loss = [e for e in valid if e.actual_exit_reason == "STOP_LOSS"]
        if len(valid) >= min_evals:
            sl_rate = len(stop_loss) / len(valid)
            if sl_rate > stop_loss_rate:
                findings.append(
                    BiasFinding(
                        bias_type="STOP_LOSS_FREQUENCY",
                        severity="MEDIUM",
                        description=(
                            f"Stop-loss exit rate is {sl_rate:.1%} (n={len(stop_loss)}), "
                            f"indicating potential overexposure or poor entry timing"
                        ),
                        metric_value=round(sl_rate, 4),
                        threshold=stop_loss_rate,
                    )
                )

        # --- HOLDING_TIME_MISMATCH ---
        short_holds = [
            e
            for e in valid
            if e.actual_duration_minutes is not None and e.actual_duration_minutes < 30
        ]
        if len(short_holds) >= min_duration_evals:
            short_wr = (
                sum(1 for e in short_holds if e.was_profitable) / len(short_holds)
            )
            if short_wr < short_hold_wr:
                findings.append(
                    BiasFinding(
                        bias_type="HOLDING_TIME_MISMATCH",
                        severity="LOW",
                        description=(
                            f"Short holds (<30min, n={len(short_holds)}) have win rate "
                            f"{short_wr:.1%}, suggesting poor short-duration trade selection"
                        ),
                        metric_value=round(short_wr, 4),
                        threshold=short_hold_wr,
                    )
                )

        return findings
