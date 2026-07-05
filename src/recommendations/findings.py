from __future__ import annotations

from typing import Sequence

from src.evaluation.models import DecisionEvaluation
from src.recommendations.models import Finding
from src.research.models import ResearchReport


class FindingsEngine:
    """Extracts immutable Finding objects from a ResearchReport + evaluations.
    Pure function — no I/O, no side effects."""

    @staticmethod
    def extract(
        report: ResearchReport,
        evaluations: Sequence[DecisionEvaluation],
    ) -> list[Finding]:
        findings: list[Finding] = []

        if report.status != "COMPLETE" or report.sample_size == 0:
            return findings

        valid_ids = [e.evaluation_id for e in evaluations if e.evaluation_id]

        # ── From bias findings ──
        for bias in report.bias_findings:
            supporting = FindingsEngine._evidence_ids_for_bias(
                bias.bias_type, evaluations
            )
            findings.append(
                Finding(
                    category=bias.bias_type,
                    description=bias.description,
                    supporting_metrics={
                        "metric_value": bias.metric_value,
                        "threshold": bias.threshold,
                        "severity": bias.severity,
                    },
                    evidence_ids=supporting if supporting else valid_ids,
                    severity=bias.severity,
                )
            )

        # ── From observations ──
        for obs in report.observations:
            findings.append(
                Finding(
                    category=obs.category,
                    description=obs.observation,
                    supporting_metrics=obs.supporting_metric,
                    evidence_ids=valid_ids,
                    severity="LOW",
                )
            )

        # ── From calibration drift (if severe) ──
        for cal in report.confidence_calibration:
            if cal.calibration_error > 0.15 and cal.sample_size >= 10:
                findings.append(
                    Finding(
                        category="CALIBRATION_DRIFT",
                        description=(
                            f"Confidence bucket {cal.bucket_label} has calibration error "
                            f"{cal.calibration_error:.1%} (win rate {cal.win_rate:.1%} vs midpoint {cal.midpoint:.1%})"
                        ),
                        supporting_metrics={
                            "bucket": cal.bucket_label,
                            "calibration_error": cal.calibration_error,
                            "win_rate": cal.win_rate,
                            "midpoint": cal.midpoint,
                            "sample_size": cal.sample_size,
                        },
                        evidence_ids=valid_ids,
                        severity="HIGH" if cal.calibration_error > 0.25 else "MEDIUM",
                    )
                )

        # ── From regime analysis ──
        for regime in report.regime_analysis:
            if regime.calibration_error > 0.15 and regime.sample_size >= 10:
                findings.append(
                    Finding(
                        category="REGIME_INEFFECTIVENESS",
                        description=(
                            f"Regime {regime.source} has calibration error "
                            f"{regime.calibration_error:.1%} (win rate {regime.win_rate:.1%} vs avg confidence {regime.avg_confidence:.1%})"
                        ),
                        supporting_metrics={
                            "source": regime.source,
                            "calibration_error": regime.calibration_error,
                            "win_rate": regime.win_rate,
                            "avg_confidence": regime.avg_confidence,
                            "sample_size": regime.sample_size,
                        },
                        evidence_ids=valid_ids,
                        severity="HIGH" if regime.calibration_error > 0.25 else "MEDIUM",
                    )
                )

        return findings

    @staticmethod
    def _evidence_ids_for_bias(
        bias_type: str,
        evaluations: Sequence[DecisionEvaluation],
    ) -> list[str]:
        if bias_type == "OVERCONFIDENCE":
            return [
                e.evaluation_id
                for e in evaluations
                if e.llm_confidence >= 0.5 and e.was_profitable is False
            ]
        elif bias_type == "UNDERCONFIDENCE":
            return [
                e.evaluation_id
                for e in evaluations
                if e.llm_confidence < 0.5 and e.was_profitable is True
            ]
        elif bias_type == "LONG_BIAS":
            return [
                e.evaluation_id
                for e in evaluations
                if e.llm_action == "BUY" and e.was_profitable is not None
            ]
        elif bias_type == "STOP_LOSS_FREQUENCY":
            return [
                e.evaluation_id
                for e in evaluations
                if e.actual_exit_reason == "STOP_LOSS"
            ]
        elif bias_type == "HOLDING_TIME_MISMATCH":
            return [
                e.evaluation_id
                for e in evaluations
                if e.actual_duration_minutes is not None
                and e.actual_duration_minutes < 30
            ]
        elif bias_type == "EVIDENCE_TIER_INEFFECTIVENESS":
            return [
                e.evaluation_id
                for e in evaluations
                if e.evidence_source in ("EXACT", "COLD_START")
            ]
        return [e.evaluation_id for e in evaluations if e.evaluation_id]
