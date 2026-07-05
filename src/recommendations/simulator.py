from __future__ import annotations

from typing import Sequence

from src.evaluation.models import DecisionEvaluation
from src.recommendations.models import Intervention, SimulationResult
from src.research.statistics import compute_sharpe_ratio


class RecommendationSimulator:
    """Counterfactual replay of DecisionEvaluation records.
    Pure function — no I/O, no side effects."""

    @staticmethod
    def simulate(
        intervention: Intervention,
        evaluations: Sequence[DecisionEvaluation],
    ) -> SimulationResult:
        valid = [
            e for e in evaluations
            if e.was_profitable is not None and e.actual_pnl is not None
        ]
        if not valid:
            return SimulationResult(intervention_id=intervention.intervention_id)

        param_id = intervention.parameter_id
        threshold = intervention.recommended_value

        # Apply the intervention: filter evaluations that pass
        filtered = [
            e for e in valid
            if RecommendationSimulator._passes(e, param_id, threshold)
        ]
        excluded = [
            e for e in valid
            if not RecommendationSimulator._passes(e, param_id, threshold)
        ]

        if len(filtered) < 3:
            return SimulationResult(intervention_id=intervention.intervention_id)

        # Current (all valid) metrics
        current_sharpe = compute_sharpe_ratio(
            [1.0 if e.was_profitable else 0.0 for e in valid]
        )
        current_wr = sum(1 for e in valid if e.was_profitable) / len(valid)
        current_profit = sum(e.actual_pnl for e in valid if e.was_profitable)
        current_loss = abs(sum(e.actual_pnl for e in valid if not e.was_profitable))
        current_pf = current_profit / current_loss if current_loss > 0 else float("inf")
        cumulative: list[float] = []
        running = 0.0
        for e in valid:
            running += e.actual_pnl
            cumulative.append(running)
        from src.research.statistics import compute_max_drawdown
        current_dd = compute_max_drawdown(cumulative)

        # Simulated (filtered) metrics
        simulated_sharpe = compute_sharpe_ratio(
            [1.0 if e.was_profitable else 0.0 for e in filtered]
        )
        simulated_wr = sum(1 for e in filtered if e.was_profitable) / len(filtered)
        sim_profit = sum(e.actual_pnl for e in filtered if e.was_profitable)
        sim_loss = abs(sum(e.actual_pnl for e in filtered if not e.was_profitable))
        sim_pf = sim_profit / sim_loss if sim_loss > 0 else float("inf")
        sim_cumulative: list[float] = []
        sim_running = 0.0
        for e in filtered:
            sim_running += e.actual_pnl
            sim_cumulative.append(sim_running)
        sim_dd = compute_max_drawdown(sim_cumulative)

        # Trade frequency change
        freq_change = ((len(filtered) - len(valid)) / len(valid)) * 100.0

        return SimulationResult(
            intervention_id=intervention.intervention_id,
            expected_sharpe_delta=round(simulated_sharpe - current_sharpe, 4),
            expected_win_rate_delta=round(simulated_wr - current_wr, 4),
            expected_profit_factor_delta=round(sim_pf - current_pf, 4) if sim_pf != float("inf") and current_pf != float("inf") else 0.0,
            expected_trade_frequency_change_pct=round(freq_change, 2),
            expected_max_drawdown_change=round(sim_dd - current_dd, 4),
            simulated_sample_size=len(filtered),
            counterfactual_wins=sum(1 for e in filtered if e.was_profitable),
            counterfactual_losses=sum(1 for e in filtered if not e.was_profitable),
        )

    @staticmethod
    def _passes(
        evaluation: DecisionEvaluation,
        parameter_id: str,
        threshold: float,
    ) -> bool:
        try:
            t = float(threshold)
        except (TypeError, ValueError):
            return True
        if parameter_id == "risk.min_llm_confidence":
            return evaluation.llm_confidence >= t
        return True
