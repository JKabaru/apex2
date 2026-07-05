from __future__ import annotations

from typing import Sequence

import structlog

from src.evaluation.models import DecisionEvaluation
from src.learning.config_catalog import ConfigurationCatalog
from src.recommendations.findings import FindingsEngine
from src.recommendations.interventions import InterventionEngine
from src.recommendations.models import Finding, Intervention, Recommendation
from src.recommendations.simulator import RecommendationSimulator
from src.recommendations.store import ConfigurationStore
from src.research.models import ResearchReport

logger = structlog.get_logger("recommendation_engine")


class RecommendationEngine:
    """Orchestrates the three-stage pipeline:
    Findings → Interventions → Recommendations (simulated).
    Never mutates config.toml or live state."""

    def __init__(
        self,
        catalog: ConfigurationCatalog,
        store: ConfigurationStore,
    ):
        self._catalog = catalog
        self._store = store
        self._findings_engine = FindingsEngine()
        self._intervention_engine = InterventionEngine(catalog)
        self._simulator = RecommendationSimulator()

    def generate(
        self,
        report: ResearchReport,
        evaluations: Sequence[DecisionEvaluation],
    ) -> tuple[list[Finding], list[Intervention], list[Recommendation]]:
        logger.info(
            "Recommendation generation started",
            report_id=report.report_id,
            sample_size=report.sample_size,
        )

        # Stage 1: Extract findings
        findings = self._findings_engine.extract(report, evaluations)
        logger.info("Findings extracted", count=len(findings))

        for finding in findings:
            self._store.save_finding(finding)

        # Stage 2: Discover interventions
        interventions: list[Intervention] = []
        for finding in findings:
            candidates = self._intervention_engine.discover(finding, evaluations)
            for c in candidates:
                self._store.save_intervention(c)
            interventions.extend(candidates)

        logger.info(
            "Interventions discovered",
            count=len(interventions),
            from_findings=len(findings),
        )

        # Stage 3: Simulate → Recommendations
        recommendations: list[Recommendation] = []
        for intervention in interventions:
            sim = self._simulator.simulate(intervention, evaluations)

            # Determine confidence tier
            tier = self._compute_confidence_tier(
                intervention, sim
            )

            improvement_pct = (
                sim.expected_sharpe_delta * 100.0
                if intervention.parameter_id != "execution.sizing_value"
                else sim.expected_win_rate_delta * 100.0
            )

            rec = Recommendation(
                intervention_id=intervention.intervention_id,
                simulation_result=sim,
                why=(
                    f"{intervention.reasoning}. "
                    f"Changing {intervention.parameter_id} from "
                    f"{intervention.current_value} to {intervention.recommended_value}."
                ),
                risk_if_wrong=self._compute_risk_if_wrong(intervention, sim),
                expected_improvement_pct=round(improvement_pct, 2),
                confidence_tier=tier,
                evidence_strength=intervention.evidence_strength,
                evidence_quality=intervention.evidence_quality,
                status="SIMULATED",
            )
            self._store.save_recommendation(rec)
            recommendations.append(rec)

        logger.info(
            "Recommendations generated",
            count=len(recommendations),
        )

        return findings, interventions, recommendations

    @staticmethod
    def _compute_confidence_tier(
        intervention: Intervention,
        sim_result: Recommendation.simulation_result,  # type: ignore
    ) -> str:
        from src.recommendations.statistics import determine_confidence_tier

        strength = (
            intervention.evidence_strength.effect_size,
            intervention.evidence_strength.magnitude_label,
            intervention.evidence_strength.ci_low,
            intervention.evidence_strength.ci_high,
        )
        quality = (
            intervention.evidence_quality.information_weight_score,
            intervention.evidence_quality.sample_size,
            intervention.evidence_quality.cross_regime_agreement,
            intervention.evidence_quality.consistency_score,
            intervention.evidence_quality.trustworthiness,
        )
        return determine_confidence_tier(strength, quality)

    @staticmethod
    def _compute_risk_if_wrong(
        intervention: Intervention,
        sim_result: Recommendation.simulation_result,  # type: ignore
    ) -> str:
        risks: list[str] = []
        trade_change = sim_result.expected_trade_frequency_change_pct
        if trade_change < -10:
            risks.append(f"May reduce trade frequency by {abs(trade_change):.0f}%")
        if sim_result.expected_max_drawdown_change > 0:
            risks.append(
                f"May increase max drawdown by "
                f"{sim_result.expected_max_drawdown_change:.1%}"
            )
        if sim_result.simulated_sample_size < 10:
            risks.append("Simulation based on very small sample")
        if not risks:
            risks.append("No significant risks detected in simulation")
        return "; ".join(risks)
