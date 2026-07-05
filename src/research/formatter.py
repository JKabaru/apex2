from __future__ import annotations

from src.research.models import ResearchReport


class ResearchFormatter:
    """Converts a ResearchReport to human-readable Markdown."""

    @staticmethod
    def to_markdown(report: ResearchReport) -> str:
        lines: list[str] = []
        lines.append(f"# Research Report — v{report.evaluation_version}")
        lines.append(f"Generated: {report.generated_at.isoformat()}")
        lines.append(f"Status: {report.status}")
        lines.append("")

        if report.status == "INSUFFICIENT_DATA":
            lines.append(
                f"**Insufficient data for analysis.** "
                f"Sample size: {report.sample_size} (minimum required: 30)."
            )
            if report.skipped_records_count > 0:
                lines.append(f"Skipped records: {report.skipped_records_count}")
            return "\n".join(lines)

        lines.append("## Overview")
        lines.append(f"- Analysis window: {report.analysis_window}")
        lines.append(f"- Sample size: {report.sample_size}")
        lines.append(f"- Skipped records: {report.skipped_records_count}")
        lines.append("")

        lines.append("## Confidence Calibration")
        lines.append(
            "| Bucket | N | Wins | Win Rate | CI (95%) | Cal Error |"
        )
        lines.append(
            "|--------|---|------|----------|----------|-----------|"
        )
        for c in report.confidence_calibration:
            ci = f"{c.wilson_ci_low:.1%}-{c.wilson_ci_high:.1%}"
            lines.append(
                f"| {c.bucket_label} | {c.sample_size} | {c.wins} | "
                f"{c.win_rate:.1%} | {ci} | {c.calibration_error:.1%} |"
            )
        lines.append("")

        lines.append("## Regime Analysis")
        lines.append(
            "| Source | N | Win Rate | Avg Confidence | Cal Error |"
        )
        lines.append(
            "|--------|---|----------|----------------|-----------|"
        )
        for r in report.regime_analysis:
            lines.append(
                f"| {r.source} | {r.sample_size} | {r.win_rate:.1%} | "
                f"{r.avg_confidence:.1%} | {r.calibration_error:.1%} |"
            )
        lines.append("")

        lines.append("## Risk Analysis")
        risk = report.risk_analysis
        if risk:
            lines.append(
                f"- Mean drawdown: {risk.get('mean_drawdown', 'N/A'):.2%}"
            )
            lines.append(
                f"- Max drawdown: {risk.get('max_drawdown', 'N/A'):.2%}"
            )
            lines.append(
                f"- Median drawdown: {risk.get('median_drawdown', 'N/A'):.2%}"
            )
            lines.append(
                f"- Drawdown P10: {risk.get('drawdown_p10', 'N/A'):.2%}"
            )
            lines.append(
                f"- Drawdown P90: {risk.get('drawdown_p90', 'N/A'):.2%}"
            )
            quartiles = risk.get("win_rate_by_drawdown_quartile", [])
            if quartiles:
                lines.append(
                    "- Win rate by drawdown quartile: "
                    + ", ".join(
                        f"Q{i+1}: {wr:.1%}"
                        for i, wr in enumerate(quartiles)
                    )
                )
        lines.append("")

        lines.append("## Holding Analysis")
        holding = report.holding_analysis
        if holding:
            lines.append(
                f"- Mean duration: {holding.get('mean_duration_minutes', 'N/A')} min"
            )
            lines.append(
                f"- Median duration: {holding.get('median_duration_minutes', 'N/A')} min"
            )
            lines.append(
                f"- IQR: {holding.get('duration_iqr_minutes', 'N/A')} min"
            )
            lines.append(
                f"- P25: {holding.get('duration_p25', 'N/A')} min"
            )
            lines.append(
                f"- P75: {holding.get('duration_p75', 'N/A')} min"
            )
            quartiles = holding.get("win_rate_by_duration_quartile", [])
            if quartiles:
                lines.append(
                    "- Win rate by duration quartile: "
                    + ", ".join(
                        f"Q{i+1}: {wr:.1%}"
                        for i, wr in enumerate(quartiles)
                    )
                )
        lines.append("")

        lines.append("## Overall Metrics")
        overall = report.overall_metrics
        if overall:
            lines.append(
                f"- Total trades: {overall.get('total_trades', 'N/A')}"
            )
            lines.append(
                f"- Profitable: {overall.get('profitable_trades', 'N/A')}"
            )
            lines.append(
                f"- Win rate: {overall.get('overall_win_rate', 'N/A'):.1%}"
            )
            wr_lo = overall.get("win_rate_ci_95_low", "N/A")
            wr_hi = overall.get("win_rate_ci_95_high", "N/A")
            lines.append(
                f"- Win rate CI (95%): {wr_lo:.1%} – {wr_hi:.1%}"
            )
            lines.append(
                f"- Sharpe ratio: {overall.get('sharpe_ratio', 'N/A')}"
            )
            lines.append(
                f"- Profit factor: {overall.get('profit_factor', 'N/A')}"
            )
            lines.append(
                f"- Max drawdown (cumulative): {overall.get('max_drawdown_cumulative_pnl', 'N/A'):.2%}"
            )
            gp = overall.get("gross_profit", 0)
            gl = overall.get("gross_loss", 0)
            lines.append(f"- Gross P&L: {gp:.2f} / {gl:.2f}")
        lines.append("")

        if report.bias_findings:
            lines.append("## Bias Findings")
            for b in report.bias_findings:
                lines.append(
                    f"- **[{b.severity}]** {b.bias_type}: {b.description}"
                )
            lines.append("")

        if report.observations:
            lines.append("## Observations")
            for o in report.observations:
                lines.append(f"- **{o.category}**: {o.observation}")
            lines.append("")

        return "\n".join(lines)
