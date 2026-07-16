from __future__ import annotations

import statistics
from typing import Sequence

from src.evaluation.models import DecisionEvaluation
from src.research.models import (
    CalibrationSummary,
    ImprovementObservation,
    RegimeBreakdown,
    ResearchReport,
)
from src.research.patterns import PatternDiscoveryEngine
from src.research.statistics import (
    compute_max_drawdown,
    compute_median,
    compute_percentile,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_wilson_interval,
)

class ResearchAnalyzer:
    """Pure function — no I/O, no side effects. Operates on list[DecisionEvaluation] only."""

    @staticmethod
    def analyze(
        evaluations: Sequence[DecisionEvaluation],
        version: str = "1.0",
        min_sample_size: int = 30,
        pattern_config: dict | None = None,
        observation_config: dict | None = None,
    ) -> ResearchReport:
        valid = [
            e
            for e in evaluations
            if e.was_profitable is not None
        ]
        skipped = len(evaluations) - len(valid)

        if not valid:
            return ResearchReport(
                evaluation_version=version,
                status="INSUFFICIENT_DATA",
                sample_size=0,
                skipped_records_count=skipped,
            )

        timestamps = [e.created_at for e in valid]
        analysis_window = (
            f"{min(timestamps).isoformat()} / {max(timestamps).isoformat()}"
        )

        if len(valid) < min_sample_size:
            return ResearchReport(
                evaluation_version=version,
                status="INSUFFICIENT_DATA",
                analysis_window=analysis_window,
                sample_size=len(valid),
                skipped_records_count=skipped,
            )

        calibration = ResearchAnalyzer._compute_calibration(valid)
        regime = ResearchAnalyzer._compute_regime_analysis(valid)
        risk = ResearchAnalyzer._compute_risk_analysis(valid)
        holding = ResearchAnalyzer._compute_holding_analysis(valid)
        overall = ResearchAnalyzer._compute_overall_metrics(valid)
        pattern_kwargs = pattern_config or {}
        biases = PatternDiscoveryEngine.discover_biases(valid, **pattern_kwargs)
        observations = ResearchAnalyzer._generate_observations(
            valid, calibration, overall, observation_config
        )

        return ResearchReport(
            evaluation_version=version,
            status="COMPLETE",
            analysis_window=analysis_window,
            sample_size=len(valid),
            skipped_records_count=skipped,
            confidence_calibration=calibration,
            regime_analysis=regime,
            risk_analysis=risk,
            holding_analysis=holding,
            overall_metrics=overall,
            bias_findings=biases,
            observations=observations,
        )

    @staticmethod
    def _compute_calibration(
        valid: Sequence[DecisionEvaluation],
    ) -> list[CalibrationSummary]:
        buckets: list[CalibrationSummary] = []
        bin_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in bin_edges:
            bucket = [
                e
                for e in valid
                if low <= e.llm_confidence < high
                or (high == 1.0 and e.llm_confidence == 1.0)
            ]
            n = len(bucket)
            if n == 0:
                continue
            wins = sum(1 for e in bucket if e.was_profitable)
            win_rate = wins / n
            midpoint = (low + high) / 2
            cal_error = abs(win_rate - midpoint)
            ci_low, ci_high = compute_wilson_interval(wins, n)
            buckets.append(
                CalibrationSummary(
                    bucket_label=f"{low:.1f}-{high:.1f}",
                    midpoint=midpoint,
                    low=low,
                    high=high,
                    sample_size=n,
                    wins=wins,
                    win_rate=round(win_rate, 4),
                    calibration_error=round(cal_error, 4),
                    wilson_ci_low=round(ci_low, 4),
                    wilson_ci_high=round(ci_high, 4),
                )
            )
        return buckets

    @staticmethod
    def _compute_regime_analysis(
        valid: Sequence[DecisionEvaluation],
    ) -> list[RegimeBreakdown]:
        sources: dict[str, list[DecisionEvaluation]] = {}
        for e in valid:
            sources.setdefault(e.evidence_source, []).append(e)
        regimes: list[RegimeBreakdown] = []
        for source in sorted(sources):
            group = sources[source]
            n = len(group)
            wins = sum(1 for e in group if e.was_profitable)
            win_rate = wins / n
            avg_conf = statistics.mean(e.llm_confidence for e in group)
            cal_error = abs(win_rate - avg_conf)
            regimes.append(
                RegimeBreakdown(
                    source=source,
                    sample_size=n,
                    win_rate=round(win_rate, 4),
                    avg_confidence=round(avg_conf, 4),
                    calibration_error=round(cal_error, 4),
                )
            )
        return regimes

    @staticmethod
    def _compute_risk_analysis(
        valid: Sequence[DecisionEvaluation],
    ) -> dict:
        drawdowns = [e.actual_max_drawdown for e in valid]
        if not drawdowns:
            return {}
        sorted_dd = sorted(drawdowns)
        mean_dd = statistics.mean(sorted_dd)
        max_dd_value = max(sorted_dd)
        med_dd = compute_median(sorted_dd)
        p10_dd = compute_percentile(sorted_dd, 10)
        p90_dd = compute_percentile(sorted_dd, 90)

        n_valid = len(valid)
        q_size = max(1, n_valid // 4)
        sorted_by_dd = sorted(valid, key=lambda e: e.actual_max_drawdown)
        quartile_wr: list[float] = []
        for i in range(4):
            start = i * q_size
            end = start + q_size if i < 3 else n_valid
            quartile = sorted_by_dd[start:end]
            if quartile:
                wr = sum(1 for e in quartile if e.was_profitable) / len(quartile)
                quartile_wr.append(round(wr, 4))

        return {
            "mean_drawdown": round(mean_dd, 4),
            "max_drawdown": round(max_dd_value, 4),
            "median_drawdown": round(med_dd, 4),
            "drawdown_p10": round(p10_dd, 4),
            "drawdown_p90": round(p90_dd, 4),
            "win_rate_by_drawdown_quartile": quartile_wr,
        }

    @staticmethod
    def _compute_holding_analysis(
        valid: Sequence[DecisionEvaluation],
    ) -> dict:
        valid_with_dur = [
            e for e in valid if e.actual_duration_minutes is not None
        ]
        if not valid_with_dur:
            return {}
        durations = [e.actual_duration_minutes for e in valid_with_dur]  # type: ignore
        sorted_dur = sorted(durations)
        mean_dur = statistics.mean(sorted_dur)
        med_dur = compute_median(sorted_dur)
        p25 = compute_percentile(sorted_dur, 25)
        p75 = compute_percentile(sorted_dur, 75)
        iqr = p75 - p25

        n = len(sorted_dur)
        q_size = max(1, n // 4)
        sorted_by_dur = sorted(valid_with_dur, key=lambda e: e.actual_duration_minutes)  # type: ignore
        quartile_wr: list[float] = []
        for i in range(4):
            start = i * q_size
            end = start + q_size if i < 3 else n
            quartile = sorted_by_dur[start:end]
            if quartile:
                wr = sum(1 for e in quartile if e.was_profitable) / len(quartile)
                quartile_wr.append(round(wr, 4))

        return {
            "mean_duration_minutes": round(mean_dur, 2),
            "median_duration_minutes": round(med_dur, 2),
            "duration_iqr_minutes": round(iqr, 2),
            "duration_p25": round(p25, 2),
            "duration_p75": round(p75, 2),
            "win_rate_by_duration_quartile": quartile_wr,
        }

    @staticmethod
    def _compute_overall_metrics(
        valid: Sequence[DecisionEvaluation],
    ) -> dict:
        total = len(valid)
        profitable = sum(1 for e in valid if e.was_profitable)
        win_rate = profitable / total if total > 0 else 0.0

        returns = [1.0 if e.was_profitable else 0.0 for e in valid]
        sharpe = compute_sharpe_ratio(returns)

        gross_profit = sum(
            e.actual_pnl
            for e in valid
            if e.was_profitable and e.actual_pnl is not None
        )
        gross_loss = sum(
            e.actual_pnl
            for e in valid
            if not e.was_profitable and e.actual_pnl is not None
        )
        abs_gross_loss = abs(gross_loss)
        profit_factor = compute_profit_factor(gross_profit, abs_gross_loss)

        cumulative = 0.0
        equity: list[float] = []
        for e in valid:
            if e.actual_pnl is not None:
                cumulative += e.actual_pnl
                equity.append(cumulative)
        max_dd = compute_max_drawdown(equity) if equity else 0.0

        wr_ci_low, wr_ci_high = compute_wilson_interval(profitable, total)

        return {
            "total_trades": total,
            "profitable_trades": profitable,
            "overall_win_rate": round(win_rate, 4),
            "win_rate_ci_95_low": round(wr_ci_low, 4),
            "win_rate_ci_95_high": round(wr_ci_high, 4),
            "sharpe_ratio": round(sharpe, 4),
            "profit_factor": (
                round(profit_factor, 4) if profit_factor != float("inf") else "inf"
            ),
            "max_drawdown_cumulative_pnl": round(max_dd, 4),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }

    @staticmethod
    def _generate_observations(
        valid: Sequence[DecisionEvaluation],
        calibration: list[CalibrationSummary],
        overall: dict,
        observation_config: dict | None = None,
    ) -> list[ImprovementObservation]:
        cfg = observation_config or {}
        cal_threshold = cfg.get("calibration_drift_threshold", 0.15)
        low_wr = cfg.get("low_win_rate", 0.4)
        high_wr = cfg.get("high_win_rate", 0.7)
        small_sample = cfg.get("small_sample_threshold", 100)

        obs: list[ImprovementObservation] = []

        if calibration:
            max_error = max(c.calibration_error for c in calibration)
            worst_bucket = max(calibration, key=lambda c: c.calibration_error)
            if max_error > cal_threshold:
                obs.append(
                    ImprovementObservation(
                        category="CALIBRATION_DRIFT",
                        observation=f"Worst calibration error is {max_error:.1%} in bucket {worst_bucket.bucket_label} (n={worst_bucket.sample_size})",
                        supporting_metric={
                            "max_calibration_error": round(max_error, 4),
                            "worst_bucket": worst_bucket.bucket_label,
                        },
                    )
                )

        wr = overall.get("overall_win_rate", 0)
        if wr < low_wr:
            obs.append(
                ImprovementObservation(
                    category="LOW_WIN_RATE",
                    observation=f"Overall win rate is {wr:.1%}, below {low_wr:.0%} threshold",
                    supporting_metric={"overall_win_rate": round(wr, 4)},
                )
            )
        elif wr > high_wr:
            obs.append(
                ImprovementObservation(
                    category="HIGH_WIN_RATE",
                    observation=f"Overall win rate is {wr:.1%}, above {high_wr:.0%} threshold",
                    supporting_metric={"overall_win_rate": round(wr, 4)},
                )
            )

        if len(valid) < small_sample:
            obs.append(
                ImprovementObservation(
                    category="SMALL_SAMPLE",
                    observation=f"Sample size of {len(valid)} is below {small_sample}; results may not be statistically significant",
                    supporting_metric={"sample_size": len(valid)},
                )
            )

        return obs
