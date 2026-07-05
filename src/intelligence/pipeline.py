from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from src.intelligence.formatter import ExperienceEvidenceFormatter
from src.intelligence.models import (
    BiasSummary,
    EvidenceProvenance,
    ExperienceEvidence,
    PromptContext,
    RepresentativeExperience,
)
from src.intelligence.patterns import PatternExtractor
from src.intelligence.policies import AnalysisPolicy
from src.intelligence.statistics import RobustCalculator
from src.intelligence.validator import EvidenceValidator
from src.retrieval.models import RetrievalRecord
from src.retrieval.report import RankedResult, RetrievalReport


class ExperienceIntelligencePipeline:
    """7-stage orchestration pipeline.

    Flow:
      RetrievalReport + AnalysisPolicy
      → Metric Extraction
      → Statistical Computation (RobustCalculator)
      → Pattern Extraction (PatternExtractor)
      → Bias Computation
      → Representative Selection
      → Evidence Validation (EvidenceValidator)
      → ExperienceEvidence

    The pipeline NEVER imports LearningManifest, LearningCorpus,
    DuckDB, SQL, or JSON paths.
    """

    def __init__(self) -> None:
        self._statistics = RobustCalculator()
        self._patterns = PatternExtractor()
        self._validator = EvidenceValidator()
        self._formatter = ExperienceEvidenceFormatter()
        self._version = "4.5.0"

    # ── public API ──────────────────────────────────────────────────────

    def process(
        self,
        report: RetrievalReport,
        policy: AnalysisPolicy | None = None,
    ) -> ExperienceEvidence:
        if policy is None:
            from src.intelligence.policies import AnalysisPolicy as DefaultPolicy
            policy = DefaultPolicy()

        source_hash = _hash_report(report)

        # Stage 1 — extract records from RankedResult
        records = [r.record for r in report.results]

        # Stage 1a — filter by minimum integrity
        filtered = [r for r in records if r.integrity_score >= policy.minimum_integrity]

        if len(filtered) < policy.minimum_sample_size:
            return ExperienceEvidence(
                sample_size=len(records),
                is_sufficient=False,
                provenance=EvidenceProvenance(
                    analysis_version=self._version,
                    source_report_hash=source_hash,
                ),
            )

        # Stage 2 — extract metric lists
        pnl_values = _extract_float(filtered, "pnl_atr_multiple")
        mae_values = _extract_float(filtered, "mae_atr_multiple")
        mfe_values = _extract_float(filtered, "mfe_atr_multiple")
        bars_values = _extract_float(filtered, "bars_held")

        # Stage 3 — statistical computation (domain-agnostic)
        pnl_cleaned, pnl_outliers = self._statistics.apply_outlier_policy(
            pnl_values, policy.outlier_policy,
        )
        pnl_dist = self._statistics.compute_distribution(pnl_cleaned)
        mae_dist = self._statistics.compute_distribution(mae_values)
        mfe_dist = self._statistics.compute_distribution(mfe_values)
        bars_dist = self._statistics.compute_distribution(bars_values)

        # win rate from cleaned PnL
        wins = sum(1 for v in pnl_cleaned if v > 0)
        win_rate = wins / len(pnl_cleaned) * 100.0 if pnl_cleaned else 0.0

        # Stage 4 — pattern extraction (generic predicates)
        success_patterns = self._patterns.extract_patterns(
            filtered, "trend_regime",
            lambda r: r.pnl_atr_multiple is not None and r.pnl_atr_multiple > 0,
            policy,
        )
        failure_patterns = self._patterns.extract_patterns(
            filtered, "trend_regime",
            lambda r: r.pnl_atr_multiple is not None and r.pnl_atr_multiple <= 0,
            policy,
        )

        # Stage 5 — bias summary
        bias = _compute_bias(filtered)

        # Stage 6 — representative selection (4-tier tie-breaker)
        reps = _select_representatives(report.results, policy.representative_count)

        # Stage 7 — validation
        avg_integrity = (
            sum(r.integrity_score for r in filtered) / len(filtered)
            if filtered
            else 0
        )
        computed_metrics = {
            "sample_size": len(pnl_cleaned),
            "avg_integrity": avg_integrity,
            "pnl_cv": pnl_dist.get("cv", 0.0),
        }
        validation = self._validator.validate(computed_metrics, policy)

        # overall confidence
        overall_confidence = _compute_confidence(
            pnl_dist, len(pnl_cleaned), avg_integrity, policy,
        )

        evidence = ExperienceEvidence(
            sample_size=len(pnl_cleaned),
            evidence_quality=_quality_label(overall_confidence),
            outlier_count=pnl_outliers,
            is_sufficient=validation.is_sufficient,
            win_rate_pct=round(win_rate, 1),
            median_pnl_atr=round(pnl_dist.get("median", 0.0), 4),
            pnl_iqr=round(pnl_dist.get("iqr", 0.0), 4),
            p10_pnl=round(pnl_dist.get("p10", 0.0), 4),
            p90_pnl=round(pnl_dist.get("p90", 0.0), 4),
            median_duration_bars=round(bars_dist.get("median", 0.0), 1),
            duration_iqr=round(bars_dist.get("iqr", 0.0), 1),
            median_mae_atr=round(mae_dist.get("median", 0.0), 4),
            median_mfe_atr=round(mfe_dist.get("median", 0.0), 4),
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            bias_summary=bias,
            representatives=reps,
            provenance=EvidenceProvenance(
                analysis_version=self._version,
                statistics_version="1.0",
                formatter_version="1.0",
                validator_version="1.0",
                source_report_hash=source_hash,
            ),
            overall_confidence=round(overall_confidence, 4),
        )

        return evidence

    def generate_prompt_context(
        self,
        evidence: ExperienceEvidence,
    ) -> PromptContext:
        if not evidence.is_sufficient:
            from src.intelligence.templates import INSUFFICIENT_TEMPLATE
            return PromptContext(
                context_string=INSUFFICIENT_TEMPLATE,
                section_order=["insufficient"],
                template_version="1.0",
                source_evidence_hash=_hash_model(evidence),
                token_count=len(INSUFFICIENT_TEMPLATE.split()),
            )
        return self._formatter.format(evidence)


# ── module-level helpers ────────────────────────────────────────────────


def _extract_float(
    records: list[RetrievalRecord],
    field: str,
) -> list[float]:
    result: list[float] = []
    for r in records:
        val = getattr(r, field, None)
        if val is not None:
            result.append(float(val))
    return result


def _select_representatives(
    results: list[RankedResult],
    count: int,
) -> list[RepresentativeExperience]:
    """4-tier deterministic tie-breaker:
    1. similarity DESC
    2. integrity DESC
    3. created_at DESC
    4. experience_id ASC
    """
    sorted_results = sorted(
        results,
        key=lambda r: (
            -r.overall_similarity,
            -r.record.integrity_score,
            -(r.record.created_at.timestamp() if r.record.created_at else 0.0),
            r.record.experience_id,
        ),
    )
    reps: list[RepresentativeExperience] = []
    for r in sorted_results[:count]:
        reps.append(RepresentativeExperience(
            experience_id=r.record.experience_id,
            similarity_score=r.overall_similarity,
            why_selected=(
                f"Highest similarity ({r.overall_similarity:.3f}) "
                f"with high integrity ({r.record.integrity_score})"
            ),
        ))
    return reps


def _compute_bias(records: list[RetrievalRecord]) -> BiasSummary:
    symbols: dict[str, int] = {}
    timeframes: dict[str, int] = {}
    regimes: dict[str, dict[str, int]] = {
        "trend": {},
        "volatility": {},
        "correlation": {},
    }
    for r in records:
        symbols[r.symbol] = symbols.get(r.symbol, 0) + 1
        timeframes[r.timeframe] = timeframes.get(r.timeframe, 0) + 1
        if r.trend_regime:
            regimes["trend"][r.trend_regime] = (
                regimes["trend"].get(r.trend_regime, 0) + 1
            )
        if r.volatility_regime:
            regimes["volatility"][r.volatility_regime] = (
                regimes["volatility"].get(r.volatility_regime, 0) + 1
            )
        if r.correlation_regime:
            regimes["correlation"][r.correlation_regime] = (
                regimes["correlation"].get(r.correlation_regime, 0) + 1
            )
    return BiasSummary(
        symbol_distribution=symbols,
        timeframe_distribution=timeframes,
        regime_distribution=regimes,
    )


def _compute_confidence(
    pnl_dist: dict,
    n: int,
    avg_integrity: int,
    policy: AnalysisPolicy,
) -> float:
    cp = policy.confidence_policy

    sample_score = min(1.0, n / cp.min_sample_size) * 0.4

    cv = pnl_dist.get("cv", 0.0)
    cv_score = (
        max(0.0, 1.0 - cv / cp.max_coefficient_of_variation) * 0.3
        if cp.max_coefficient_of_variation > 0
        else 0.3
    )

    integ_score = min(1.0, avg_integrity / cp.min_avg_integrity) * 0.3

    return sample_score + cv_score + integ_score


def _quality_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "HIGH"
    if confidence >= 0.5:
        return "MEDIUM"
    return "LOW"


def _hash_report(report: RetrievalReport) -> str:
    dump = json.dumps(
        report.model_dump(mode="json", exclude={"generated_at", "execution_time_ms"}),
        sort_keys=True,
    )
    return hashlib.sha256(dump.encode()).hexdigest()[:16]


def _hash_model(model: BaseModel) -> str:
    dump = json.dumps(model.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(dump.encode()).hexdigest()[:16]
