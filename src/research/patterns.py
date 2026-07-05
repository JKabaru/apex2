from __future__ import annotations

import statistics
from typing import Sequence

from src.evaluation.models import DecisionEvaluation
from src.research.models import BiasFinding


class PatternDiscoveryEngine:
    """Deterministic bias detection over a list of DecisionEvaluation objects."""

    @staticmethod
    def discover_biases(
        evaluations: Sequence[DecisionEvaluation],
    ) -> list[BiasFinding]:
        findings: list[BiasFinding] = []

        valid = [e for e in evaluations if e.was_profitable is not None]
        if len(valid) < 10:
            return findings

        # --- OVERCONFIDENCE ---
        high_conf = [e for e in valid if e.llm_confidence >= 0.5]
        if len(high_conf) >= 10:
            high_conf_wins = sum(1 for e in high_conf if e.was_profitable)
            high_conf_wr = high_conf_wins / len(high_conf)
            avg_conf = statistics.mean(e.llm_confidence for e in high_conf)
            gap = avg_conf - high_conf_wr
            if gap > 0.30:
                findings.append(
                    BiasFinding(
                        bias_type="OVERCONFIDENCE",
                        severity="HIGH",
                        description=(
                            f"High-confidence trades (n={len(high_conf)}) have win rate "
                            f"{high_conf_wr:.1%} vs avg confidence {avg_conf:.1%} (gap={gap:.1%})"
                        ),
                        metric_value=round(gap, 4),
                        threshold=0.30,
                    )
                )
            elif gap > 0.15:
                findings.append(
                    BiasFinding(
                        bias_type="OVERCONFIDENCE",
                        severity="MEDIUM",
                        description=(
                            f"High-confidence trades (n={len(high_conf)}) have win rate "
                            f"{high_conf_wr:.1%} vs avg confidence {avg_conf:.1%} (gap={gap:.1%})"
                        ),
                        metric_value=round(gap, 4),
                        threshold=0.15,
                    )
                )

        # --- UNDERCONFIDENCE ---
        low_conf = [e for e in valid if e.llm_confidence < 0.5]
        if len(low_conf) >= 10:
            low_conf_wins = sum(1 for e in low_conf if e.was_profitable)
            low_conf_wr = low_conf_wins / len(low_conf)
            if low_conf_wr > 0.60:
                findings.append(
                    BiasFinding(
                        bias_type="UNDERCONFIDENCE",
                        severity="MEDIUM",
                        description=(
                            f"Low-confidence trades (n={len(low_conf)}) have win rate "
                            f"{low_conf_wr:.1%}, suggesting the model undervalues its own predictions"
                        ),
                        metric_value=round(low_conf_wr, 4),
                        threshold=0.60,
                    )
                )

        # --- LONG_BIAS ---
        buy_trades = [e for e in valid if e.llm_action == "BUY"]
        sell_trades = [e for e in valid if e.llm_action == "SELL"]
        if len(buy_trades) >= 5 and len(sell_trades) >= 5:
            buy_wr = sum(1 for e in buy_trades if e.was_profitable) / len(buy_trades)
            sell_wr = (
                sum(1 for e in sell_trades if e.was_profitable) / len(sell_trades)
            )
            diff = abs(buy_wr - sell_wr)
            if diff > 0.20:
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
                        threshold=0.20,
                    )
                )

        # --- EVIDENCE_TIER_INEFFECTIVENESS ---
        exact = [e for e in valid if e.evidence_source == "EXACT"]
        cold = [e for e in valid if e.evidence_source == "COLD_START"]
        if len(exact) >= 5 and len(cold) >= 5:
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
        if len(valid) >= 10:
            sl_rate = len(stop_loss) / len(valid)
            if sl_rate > 0.40:
                findings.append(
                    BiasFinding(
                        bias_type="STOP_LOSS_FREQUENCY",
                        severity="MEDIUM",
                        description=(
                            f"Stop-loss exit rate is {sl_rate:.1%} (n={len(stop_loss)}), "
                            f"indicating potential overexposure or poor entry timing"
                        ),
                        metric_value=round(sl_rate, 4),
                        threshold=0.40,
                    )
                )

        # --- HOLDING_TIME_MISMATCH ---
        short_holds = [
            e
            for e in valid
            if e.actual_duration_minutes is not None and e.actual_duration_minutes < 30
        ]
        if len(short_holds) >= 5:
            short_wr = (
                sum(1 for e in short_holds if e.was_profitable) / len(short_holds)
            )
            if short_wr < 0.40:
                findings.append(
                    BiasFinding(
                        bias_type="HOLDING_TIME_MISMATCH",
                        severity="LOW",
                        description=(
                            f"Short holds (<30min, n={len(short_holds)}) have win rate "
                            f"{short_wr:.1%}, suggesting poor short-duration trade selection"
                        ),
                        metric_value=round(short_wr, 4),
                        threshold=0.40,
                    )
                )

        return findings
