from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
import structlog

from src.intelligence.formatter import ExperienceEvidenceFormatter
from src.intelligence.models import (
    BiasSummary,
    EvidenceProvenance,
    ExperienceEvidence,
    LiveTrajectory,
    PromptContext,
    RepresentativeExperience,
)
from src.intelligence.patterns import PatternExtractor
from src.intelligence.policies import AnalysisPolicy
from src.intelligence.statistics import RobustCalculator
from src.intelligence.validator import EvidenceValidator
from src.retrieval.models import RetrievalRecord
from src.retrieval.report import RankedResult, RetrievalReport

logger = structlog.get_logger("intel_pipeline")


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

        # Split into live (interim) and finalized
        live_records = [r for r in records if r.record_source == "interim"]
        finalized_records = [r for r in records if r.record_source != "interim"]

        logger.info(
            "INTEL_PIPELINE_RECORDS",
            total=len(records),
            live=len(live_records),
            finalized=len(finalized_records),
            _force_log=True,
        )

        # Extract live trajectories from interim records (bypasses integrity/size gates)
        live_trajectories = _extract_live_trajectories(live_records)

        if live_trajectories:
            symbols = [t.symbol for t in live_trajectories]
            logger.info(
                "INTEL_PIPELINE_LIVE_TRAJECTORIES",
                count=len(live_trajectories),
                symbols=symbols,
                _force_log=True,
            )

        # Stage 1a — filter finalized records by minimum integrity
        filtered = [r for r in finalized_records if r.integrity_score >= policy.minimum_integrity]

        if len(filtered) < policy.minimum_sample_size:
            return ExperienceEvidence(
                sample_size=len(records),
                is_sufficient=False,
                live_trajectories=live_trajectories,
                provenance=EvidenceProvenance(
                    analysis_version=self._version,
                    source_report_hash=source_hash,
                ),
            )

        # Stage 2 — extract metric lists from finalized records only
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

        # episode aggregation
        records_with_episodes = sum(1 for r in filtered if r.episode_count > 0)
        total_episodes = sum(r.episode_count for r in filtered)
        avg_episode_count = total_episodes / len(filtered) if filtered else 0.0

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
            avg_episode_count=round(avg_episode_count, 1),
            records_with_episodes=records_with_episodes,
            total_episodes=total_episodes,
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
            live_trajectories=live_trajectories,
        )

        return evidence

    def generate_prompt_context(
        self,
        evidence: ExperienceEvidence,
    ) -> PromptContext:
        from src.intelligence.templates import INSUFFICIENT_TEMPLATE

        live_section = _format_live_section(evidence.live_trajectories)

        if not evidence.is_sufficient and not live_section:
            return PromptContext(
                context_string=INSUFFICIENT_TEMPLATE,
                section_order=["insufficient"],
                template_version="1.0",
                source_evidence_hash=_hash_model(evidence),
                token_count=len(INSUFFICIENT_TEMPLATE.split()),
            )

        if evidence.is_sufficient:
            ctx = self._formatter.format(evidence)
        else:
            ctx = PromptContext(
                context_string=INSUFFICIENT_TEMPLATE,
                section_order=["insufficient"],
                template_version="1.0",
                source_evidence_hash=_hash_model(evidence),
                token_count=len(INSUFFICIENT_TEMPLATE.split()),
            )

        if live_section:
            parts = [ctx.context_string, live_section] if ctx.context_string else [live_section]
            combined = "\n".join(parts)
            logger.info(
                "INTEL_PIPELINE_LIVE_SECTION_APPENDED",
                live_count=len(evidence.live_trajectories),
                token_count=len(combined.split()),
                was_sufficient=evidence.is_sufficient,
                _force_log=True,
            )
            return PromptContext(
                context_string=combined,
                section_order=ctx.section_order + ["live_positions"],
                template_version=ctx.template_version,
                source_evidence_hash=ctx.source_evidence_hash,
                token_count=len(combined.split()),
                evidence_tier=ctx.evidence_tier,
                evidence_source=ctx.evidence_source,
                has_live_data=True,
            )

        return ctx


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


def _extract_live_trajectories(
    records: list[RetrievalRecord],
) -> list[LiveTrajectory]:
    trajectories: list[LiveTrajectory] = []
    now = datetime.now(timezone.utc)
    for r in records:
        if not r.created_at:
            continue
        ct = r.created_at
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        open_minutes = int((now - ct).total_seconds() / 60)
        eps_summary = _format_episodes_summary(r.evidence_episodes_summary)
        threat = _compute_threat_level(r.episode_count, open_minutes)
        trajectories.append(LiveTrajectory(
            position_id=r.position_id,
            symbol=r.symbol,
            side=r.side,
            open_duration_minutes=open_minutes,
            current_pnl_atr=r.pnl_atr_multiple,
            episodes_summary=eps_summary,
            episode_count=r.episode_count,
            threat_level=threat,
        ))
    return trajectories


def _format_episodes_summary(
    episodes: list[dict[str, Any]] | None,
) -> list[str]:
    if not episodes:
        return []
    profiles: list[str] = []
    for ep in episodes:
        sp = ep.get("state_profile", "")
        if sp:
            profiles.append(str(sp))
    return profiles


def _compute_threat_level(
    episode_count: int,
    open_minutes: int,
) -> str:
    if episode_count >= 5 or open_minutes > 360:
        return "high"
    if episode_count >= 3 or open_minutes > 120:
        return "medium"
    return "low"


def _format_live_section(
    trajectories: list[LiveTrajectory],
) -> str:
    if not trajectories:
        return ""
    from src.intelligence.templates import (
        LIVE_POSITIONS_HEADER,
        LIVE_TEMPLATE,
    )
    lines: list[str] = [LIVE_POSITIONS_HEADER]
    for t in trajectories:
        pnl_str = f"{t.current_pnl_atr:.2f}" if t.current_pnl_atr is not None else "N/A"
        ep_str = "→".join(t.episodes_summary) if t.episodes_summary else f"{t.episode_count} episodes"
        line = LIVE_TEMPLATE.format(
            symbol=t.symbol,
            side=t.side,
            duration=t.open_duration_minutes,
            pnl=pnl_str,
            eps=ep_str,
            threat=t.threat_level,
        )
        lines.append(line)
    return "\n".join(lines)
