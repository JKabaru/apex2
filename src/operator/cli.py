from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.recommendations.models import ConfigurationProfile, Recommendation

logger = structlog.get_logger("operator_cli")


def _fmt_delta(val: float) -> str:
    if val >= 0:
        return f"+{val:.0%}"
    return f"{val:.0%}"


def _build_diff(
    current: dict[str, Any],
    chosen: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    diff: list[tuple[str, Any, Any]] = []
    all_keys = set(current.keys()) | set(chosen.keys())
    for key in sorted(all_keys):
        old_val = current.get(key)
        new_val = chosen.get(key)
        if old_val != new_val:
            diff.append((key, old_val, new_val))
    return diff


def _color_value(val: Any, old_val: Any) -> str:
    if not isinstance(val, (int, float)) or not isinstance(old_val, (int, float)):
        return "cyan"
    if val > old_val:
        return "green"
    if val < old_val:
        return "yellow"
    return "cyan"


class StartupCLI:
    """Presents profile selection to the operator before a trading session begins.

    Flow:
      1. Research summary — how recent, how many recommendations
      2. Available profiles — numbered list with derived metrics
      3. Selection prompt (30s timeout → keep current)
      4. Configuration diff (if different profile chosen)
      5. Confirmation prompt (30s timeout → reject)
    """

    @staticmethod
    async def run_startup_review(
        profiles: list[ConfigurationProfile],
        recommendations: list[Recommendation],
        current_active: Optional[ConfigurationProfile],
    ) -> Optional[str]:
        """Returns chosen profile_id, or None to keep current active profile."""
        console = Console()

        # ── Research Summary ──
        rec_map = {r.recommendation_id: r for r in recommendations}
        now = datetime.utcnow()

        rec_times = [r.created_at for r in recommendations if r.created_at]
        latest_research = max(rec_times) if rec_times else None
        research_age_h = (now - latest_research).total_seconds() / 3600 if latest_research else None

        high = sum(1 for r in recommendations if r.confidence_tier == "HIGH")
        medium = sum(1 for r in recommendations if r.confidence_tier == "MEDIUM")

        summary_parts: list[tuple[str, str]] = []
        if latest_research:
            summary_parts.append(("Last Research", latest_research.strftime("%Y-%m-%d %H:%M UTC")))
        if research_age_h is not None:
            age_str = f"{research_age_h:.0f} hour{'s' if research_age_h >= 2 else ''}"
            summary_parts.append(("Research Age", age_str))
        summary_parts.append(("Evaluated Trades", "—"))
        summary_parts.append(("New Recommendations", str(len(recommendations))))
        summary_parts.append(("High Confidence", str(high)))
        summary_parts.append(("Medium Confidence", str(medium)))

        summary_table = Table.grid(padding=(0, 2))
        summary_table.add_column(style="dim", no_wrap=True)
        summary_table.add_column(style="white")
        for label, value in summary_parts:
            summary_table.add_row(f"{label}:", value)

        console.print()
        console.print(Panel.fit(summary_table, title="Research Summary", title_align="left", border_style="cyan"))

        # ── Available Profiles ──
        if not profiles:
            console.print("[yellow]No profiles found in store.[/]")
            return current_active.profile_id if current_active else None

        profile_table = Table(title="Available Profiles", title_style="bold cyan", border_style="dim")
        profile_table.add_column("#", style="bold", width=3)
        profile_table.add_column("Name", style="white", no_wrap=True)
        profile_table.add_column("Metrics", style="dim")
        profile_table.add_column("Status", style="dim", width=10)

        for i, p in enumerate(profiles, 1):
            metrics_parts: list[str] = []
            for rec_id in p.derived_from_recommendations:
                rec = rec_map.get(rec_id)
                if rec and rec.simulation_result:
                    sim = rec.simulation_result
                    if sim.expected_sharpe_delta != 0:
                        metrics_parts.append(f"Sharpe {_fmt_delta(sim.expected_sharpe_delta)}")
                    if sim.expected_win_rate_delta != 0:
                        metrics_parts.append(f"WinRate {_fmt_delta(sim.expected_win_rate_delta)}")
                    if sim.expected_max_drawdown_change != 0:
                        metrics_parts.append(f"DD {_fmt_delta(sim.expected_max_drawdown_change)}")
                    if sim.expected_trade_frequency_change_pct != 0:
                        metrics_parts.append(f"Freq {sim.expected_trade_frequency_change_pct:+.0%}")

            metrics_str = "  ".join(metrics_parts[:3]) if metrics_parts else ""
            if p.system_generated:
                name_str = f"{p.name} (system)"
            else:
                name_str = p.name

            status = "[green]Active[/]" if p.is_active else ""
            profile_table.add_row(str(i), name_str, metrics_str, status)

        console.print()
        console.print(profile_table)
        console.print()

        # ── Selection ──
        default_idx = 0
        for i, p in enumerate(profiles):
            if p.is_active:
                default_idx = i
                break

        try:
            choice = await asyncio.wait_for(
                asyncio.to_thread(input, f"Choose profile [{default_idx + 1}]: "),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, EOFError):
            logger.warning("Profile selection timed out or unavailable, keeping current active profile")
            return current_active.profile_id if current_active else (profiles[0].profile_id if profiles else None)

        choice = choice.strip()
        if choice == "":
            selected_idx = default_idx
        else:
            try:
                selected_idx = int(choice) - 1
                if selected_idx < 0 or selected_idx >= len(profiles):
                    console.print(f"[red]Invalid selection. Keeping current profile.[/]")
                    return current_active.profile_id if current_active else profiles[0].profile_id
            except ValueError:
                console.print(f"[red]Invalid input. Keeping current profile.[/]")
                return current_active.profile_id if current_active else profiles[0].profile_id

        chosen = profiles[selected_idx]

        # ── Diff + Confirmation ──
        if current_active and chosen.profile_id != current_active.profile_id:
            diff = _build_diff(current_active.resolved_configuration, chosen.resolved_configuration)
            if diff:
                diff_table = Table(
                    title=f"Configuration Changes: '{current_active.name}' → '{chosen.name}'",
                    title_style="bold yellow",
                    border_style="dim",
                )
                diff_table.add_column("Parameter", style="bold", no_wrap=True)
                diff_table.add_column("Current", style="dim")
                diff_table.add_column("", style="bold", width=2)
                diff_table.add_column("New", style="bold")

                for param, old_val, new_val in diff:
                    color = _color_value(new_val, old_val)
                    old_str = str(old_val) if old_val is not None else "—"
                    new_str = str(new_val) if new_val is not None else "—"
                    diff_table.add_row(param, old_str, "→", f"[{color}]{new_str}[/]")

                console.print()
                console.print(diff_table)
                console.print()

            console.print(f"[bold]Activate profile '{chosen.name}'?[/]")
            try:
                confirm = await asyncio.wait_for(
                    asyncio.to_thread(input, "Proceed? (y/N): "),
                    timeout=30.0,
                )
            except (asyncio.TimeoutError, EOFError):
                logger.warning("Activation confirmation timed out or unavailable, keeping current profile")
                return current_active.profile_id if current_active else (profiles[default_idx].profile_id if profiles else None)

            if confirm.strip().lower() != "y":
                console.print("[yellow]Activation cancelled. Keeping current profile.[/]")
                return current_active.profile_id if current_active else (profiles[default_idx].profile_id if profiles else None)

        console.print()
        console.print(
            Panel.fit(
                Text.assemble(
                    ("Starting Trading Session\n", "bold green"),
                    (f"Profile: {chosen.name}", "white"),
                ),
                border_style="green",
            )
        )

        return chosen.profile_id
