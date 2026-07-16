from __future__ import annotations

import copy
import statistics
from typing import Any, Optional, Sequence

from src.evaluation.models import DecisionEvaluation
from src.learning.config_catalog import ConfigurationCatalog, ConfigurationItem
from src.recommendations.models import EvidenceQuality, EvidenceStrength, Finding, Intervention
from src.recommendations.statistics import (
    compute_evidence_quality,
    compute_evidence_strength,
    compute_information_weight,
)


class InterventionEngine:
    """Generates candidate interventions from findings.
    Maps each finding category to relevant catalog parameters,
    computes optimal values via grid sweep, and filters by
    statistical significance."""

    # Maps finding categories to parameter_ids and sweep config
    _PARAMETER_MAP: dict[str, list[dict[str, Any]]] = {
        "OVERCONFIDENCE": [
            {
                "parameter_id": "risk.min_llm_confidence",
                "metric": "win_rate",
                "steps": 10,
                "description": "Increase minimum LLM confidence to filter low-confidence losers",
            },
        ],
        "UNDERCONFIDENCE": [
            {
                "parameter_id": "risk.min_llm_confidence",
                "metric": "trade_frequency",
                "steps": 10,
                "description": "Decrease minimum LLM confidence to capture more opportunities",
            },
        ],
        "LONG_BIAS": [
            {
                "parameter_id": "risk.min_llm_confidence",
                "metric": "win_rate_gap",
                "steps": 10,
                "description": "Adjust confidence threshold to balance long/short performance",
            },
        ],
        "STOP_LOSS_FREQUENCY": [
            {
                "parameter_id": "execution.stop_loss_pct",
                "metric": "avg_loss",
                "steps": 10,
                "description": "Tighten stop loss to reduce loss frequency",
            },
            {
                "parameter_id": "execution.trailing_stop_atr_mult",
                "metric": "avg_loss",
                "steps": 8,
                "description": "Adjust ATR trailing stop multiplier for better exits",
            },
        ],
        "HOLDING_TIME_MISMATCH": [
            {
                "parameter_id": "execution.stop_loss_pct",
                "metric": "short_hold_win_rate",
                "steps": 10,
                "description": "Tighten stop to exit losing short-hold trades faster",
            },
            {
                "parameter_id": "execution.trailing_stop_atr_mult",
                "metric": "short_hold_win_rate",
                "steps": 8,
                "description": "Adjust trailing stop for better short-duration exits",
            },
        ],
        "CALIBRATION_DRIFT": [
            {
                "parameter_id": "risk.min_llm_confidence",
                "metric": "calibration_error",
                "steps": 10,
                "description": "Raise confidence floor to reduce calibration error in upper buckets",
            },
        ],
        "REGIME_INEFFECTIVENESS": [
            {
                "parameter_id": "risk.min_llm_confidence",
                "metric": "win_rate",
                "steps": 10,
                "description": "Increase confidence requirement for underperforming regime",
            },
        ],
        "LOW_WIN_RATE": [
            {
                "parameter_id": "risk.min_llm_confidence",
                "metric": "win_rate",
                "steps": 10,
                "description": "Raise minimum confidence to improve overall win rate",
            },
            {
                "parameter_id": "execution.sizing_value",
                "metric": "sharpe",
                "steps": 10,
                "description": "Reduce position size to manage risk while win rate is low",
            },
        ],
        "HIGH_WIN_RATE": [
            {
                "parameter_id": "risk.min_llm_confidence",
                "metric": "trade_frequency",
                "steps": 10,
                "description": "Lower confidence floor to capture more trades without sacrificing quality",
            },
        ],
    }

    def __init__(self, catalog: ConfigurationCatalog, metric_config: dict | None = None):
        self._catalog = catalog
        mcfg = metric_config or {}
        self._min_metric_subgroup = mcfg.get("min_metric_subgroup", 3)
        self._min_metric_losses = mcfg.get("min_metric_losses", 2)
        self._min_improvement = mcfg.get("min_improvement", 0.01)

    def discover(
        self,
        finding: Finding,
        evaluations: Sequence[DecisionEvaluation],
        min_evals: int = 5,
        min_simulation_evals: int = 3,
        min_effect_size: float = 0.2,
    ) -> list[Intervention]:
        if finding.category not in self._PARAMETER_MAP:
            return []

        interventions: list[Intervention] = []
        param_configs = self._PARAMETER_MAP[finding.category]

        relevant_evals = self._filter_evaluations(finding, evaluations)
        if len(relevant_evals) < min_evals:
            return interventions

        for config in param_configs:
            item = self._catalog.get_item(config["parameter_id"])
            if item is None or not item.learnable:
                continue

            intervention = self._build_intervention(
                finding=finding,
                item=item,
                config=config,
                evaluations=relevant_evals,
                all_evaluations=evaluations,
                min_evals=min_evals,
                min_simulation_evals=min_simulation_evals,
                min_effect_size=min_effect_size,
            )
            if intervention is not None:
                interventions.append(intervention)

        return interventions

    def _filter_evaluations(
        self,
        finding: Finding,
        evaluations: Sequence[DecisionEvaluation],
    ) -> list[DecisionEvaluation]:
        evidence_set = set(finding.evidence_ids)
        if not evidence_set:
            return [e for e in evaluations if e.was_profitable is not None]
        return [
            e for e in evaluations
            if e.evaluation_id in evidence_set and e.was_profitable is not None
        ]

    def _build_intervention(
        self,
        finding: Finding,
        item: ConfigurationItem,
        config: dict[str, Any],
        evaluations: list[DecisionEvaluation],
        all_evaluations: Sequence[DecisionEvaluation],
        min_evals: int = 5,
        min_simulation_evals: int = 3,
        min_effect_size: float = 0.2,
    ) -> Optional[Intervention]:
        current_value = item.current_default
        lo = item.minimum if item.minimum is not None else 0.0
        hi = item.maximum if item.maximum is not None else 1.0

        # Sweep over the FULL evaluation set to find optimal threshold
        full_valid = [e for e in all_evaluations if e.was_profitable is not None]
        if len(full_valid) < min_evals:
            return None

        best_value, best_metric = self._sweep_parameter(
            full_valid, item.parameter_id, config["metric"], lo, hi, config["steps"]
        )
        if best_value is None or best_value == current_value:
            return None

        # Compute current metric on full set
        current_metric = self._compute_metric_at_threshold(
            full_valid, item.parameter_id, config["metric"], current_value
        )

        improvement = best_metric - current_metric
        if improvement <= 0 and config["metric"] not in ("avg_loss", "calibration_error"):
            return None
        if improvement <= self._min_improvement:
            return None

        # Test group: trades passing the proposed threshold
        test_group = [
            e for e in full_valid
            if self._passes_threshold(e, item.parameter_id, best_value)
        ]
        # Control group: trades passing current but NOT proposed threshold
        control_group = [
            e for e in full_valid
            if self._passes_threshold(e, item.parameter_id, current_value)
            and not self._passes_threshold(e, item.parameter_id, best_value)
        ]

        test_wins = sum(1 for e in test_group if e.was_profitable)
        test_total = len(test_group)
        control_wins = sum(1 for e in control_group if e.was_profitable)
        control_total = len(control_group)

        if test_total < min_simulation_evals or control_total < min_simulation_evals:
            return None

        d, mag_label, ci_lo, ci_hi = compute_evidence_strength(
            test_wins, test_total, control_wins, control_total
        )
        if abs(d) < min_effect_size:
            return None

        supporting_ids = {e.evaluation_id for e in test_group if e.evaluation_id}
        conflicting_ids = {e.evaluation_id for e in control_group if e.evaluation_id}
        total_info_weight, _, cross_regime, consistency, trust = compute_evidence_quality(
            full_valid, supporting_ids, conflicting_ids
        )

        total_weight = sum(
            compute_information_weight(e) for e in test_group
        )

        reasoning = config["description"]

        return Intervention(
            finding_id=finding.finding_id,
            parameter_id=item.parameter_id,
            current_value=current_value,
            recommended_value=best_value,
            reasoning=reasoning,
            evidence_strength=EvidenceStrength(
                effect_size=round(d, 4),
                magnitude_label=mag_label,
                ci_low=round(ci_lo, 4),
                ci_high=round(ci_hi, 4),
            ),
            evidence_quality=EvidenceQuality(
                information_weight_score=round(total_info_weight, 4),
                sample_size=test_total + control_total,
                cross_regime_agreement=round(cross_regime, 4),
                consistency_score=round(consistency, 4),
                trustworthiness=trust,
            ),
            sample_size=test_total,
            information_weight_score=round(total_weight, 4),
            supporting_evidence_ids=list(supporting_ids),
            conflicting_evidence_ids=list(conflicting_ids),
        )

    def _compute_metric_at_threshold(
        self,
        evaluations: list[DecisionEvaluation],
        parameter_id: str,
        metric: str,
        threshold: float,
    ) -> float:
        subgroup = [
            e for e in evaluations
            if self._passes_threshold(e, parameter_id, threshold)
        ]
        if len(subgroup) < self._min_metric_subgroup:
            return 0.0

        if metric == "win_rate":
            return sum(1 for e in subgroup if e.was_profitable) / len(subgroup)
        elif metric == "trade_frequency":
            return float(len(subgroup))
        elif metric == "avg_loss":
            losses = [e for e in subgroup if not e.was_profitable]
            if len(losses) < self._min_metric_losses:
                return 0.0
            return -abs(statistics.mean([e.actual_pnl for e in losses]))
        elif metric == "short_hold_win_rate":
            short = [
                e for e in subgroup
                if e.actual_duration_minutes is not None and e.actual_duration_minutes < 30
            ]
            if len(short) < self._min_metric_subgroup:
                return 0.0
            return sum(1 for e in short if e.was_profitable) / len(short)
        elif metric == "win_rate_gap":
            buys = [e for e in subgroup if e.llm_action == "BUY"]
            sells = [e for e in subgroup if e.llm_action == "SELL"]
            buy_wr = sum(1 for e in buys if e.was_profitable) / len(buys) if len(buys) >= self._min_metric_subgroup else 0.0
            sell_wr = sum(1 for e in sells if e.was_profitable) / len(sells) if len(sells) >= self._min_metric_subgroup else 0.0
            return -abs(buy_wr - sell_wr)
        elif metric == "calibration_error":
            avg_conf = statistics.mean(e.llm_confidence for e in subgroup)
            wr = sum(1 for e in subgroup if e.was_profitable) / len(subgroup)
            return -abs(wr - avg_conf)
        elif metric == "sharpe":
            from src.research.statistics import compute_sharpe_ratio
            returns = [1.0 if e.was_profitable else 0.0 for e in subgroup]
            return compute_sharpe_ratio(returns)
        return 0.0

    def _sweep_parameter(
        self,
        evaluations: list[DecisionEvaluation],
        parameter_id: str,
        metric: str,
        lo: float,
        hi: float,
        steps: int,
    ) -> tuple[Optional[float], float]:
        best_value: Optional[float] = None
        best_metric = -float("inf")
        step_size = (hi - lo) / steps if steps > 0 else 0.0

        for i in range(steps + 1):
            candidate = lo + i * step_size
            if candidate > hi:
                candidate = hi

            subgroup = [
                e for e in evaluations
                if self._passes_threshold(e, parameter_id, candidate)
            ]
            if len(subgroup) < self._min_metric_subgroup:
                continue

            if metric == "win_rate":
                val = sum(1 for e in subgroup if e.was_profitable) / len(subgroup)
            elif metric == "trade_frequency":
                val = float(len(subgroup))
            elif metric == "avg_loss":
                losses = [e for e in subgroup if not e.was_profitable]
                val = (
                    -abs(statistics.mean([e.actual_pnl for e in losses]))
                    if len(losses) >= self._min_metric_losses else 0.0
                )
            elif metric == "short_hold_win_rate":
                short = [
                    e for e in subgroup
                    if e.actual_duration_minutes is not None
                    and e.actual_duration_minutes < 30
                ]
                val = (
                    sum(1 for e in short if e.was_profitable) / len(short)
                    if len(short) >= self._min_metric_subgroup else 0.0
                )
            elif metric == "win_rate_gap":
                buys = [e for e in subgroup if e.llm_action == "BUY"]
                sells = [e for e in subgroup if e.llm_action == "SELL"]
                buy_wr = sum(1 for e in buys if e.was_profitable) / len(buys) if len(buys) >= self._min_metric_subgroup else 0.0
                sell_wr = sum(1 for e in sells if e.was_profitable) / len(sells) if len(sells) >= self._min_metric_subgroup else 0.0
                val = -abs(buy_wr - sell_wr)
            elif metric == "calibration_error":
                if subgroup:
                    avg_conf = statistics.mean(e.llm_confidence for e in subgroup)
                    wr = sum(1 for e in subgroup if e.was_profitable) / len(subgroup)
                    val = -abs(wr - avg_conf)
                else:
                    val = 0.0
            elif metric == "sharpe":
                from src.research.statistics import compute_sharpe_ratio
                returns = [1.0 if e.was_profitable else 0.0 for e in subgroup]
                val = compute_sharpe_ratio(returns)
            else:
                val = 0.0

            if val > best_metric:
                best_metric = val
                best_value = candidate

        return best_value, best_metric

    @staticmethod
    def _passes_threshold(
        evaluation: DecisionEvaluation,
        parameter_id: str,
        threshold: Any,
    ) -> bool:
        try:
            threshold_f = float(threshold)
        except (TypeError, ValueError):
            return True

        if parameter_id == "risk.min_llm_confidence":
            return evaluation.llm_confidence >= threshold_f
        elif parameter_id == "execution.stop_loss_pct":
            return evaluation.actual_max_drawdown <= (1.0 - threshold_f)
        elif parameter_id == "execution.trailing_stop_atr_mult":
            return True
        elif parameter_id == "execution.sizing_value":
            return True
        return True
