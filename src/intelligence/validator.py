from __future__ import annotations

from pydantic import BaseModel, Field

from src.intelligence.policies import AnalysisPolicy


class ValidationResult(BaseModel, frozen=True):
    is_sufficient: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


class EvidenceValidator:
    """Pure verification gate — NEVER computes metrics.

    Receives pre-computed metrics from the pipeline and checks them
    against the AnalysisPolicy thresholds. Returns a ValidationResult
    with is_sufficient=True only when ALL constraints are met.
    """

    @staticmethod
    def validate(
        computed_metrics: dict,
        policy: AnalysisPolicy,
    ) -> ValidationResult:
        reasons: list[str] = []

        sample_size = computed_metrics.get("sample_size", 0)
        if sample_size < policy.minimum_sample_size:
            reasons.append(
                f"Sample size {sample_size} below minimum "
                f"{policy.minimum_sample_size}"
            )

        avg_integrity = computed_metrics.get("avg_integrity", 0)
        if avg_integrity < policy.confidence_policy.min_avg_integrity:
            reasons.append(
                f"Average integrity {avg_integrity} below minimum "
                f"{policy.confidence_policy.min_avg_integrity}"
            )

        cv = computed_metrics.get("pnl_cv", 0.0)
        max_cv = policy.confidence_policy.max_coefficient_of_variation
        if cv > max_cv:
            reasons.append(
                f"PnL CV {cv:.2f} exceeds maximum {max_cv}"
            )

        return ValidationResult(
            is_sufficient=len(reasons) == 0,
            rejection_reasons=reasons,
        )
