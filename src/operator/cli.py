from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import questionary
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.recommendations.models import ConfigurationProfile, LearningPolicy, MemoryWorkspace, Recommendation
from src.storage.learning.learning_corpus import LearningCorpus

logger = structlog.get_logger("operator_cli")

CHECKPOINT_PATH = "data/.active_state.json"


def write_checkpoint(profile_id: str | None, workspace_id: str | None) -> None:
    import json, os, tempfile
    tmp = CHECKPOINT_PATH + ".tmp"
    data = {
        "profile_id": profile_id,
        "workspace_id": workspace_id,
        "updated_at": datetime.utcnow().isoformat(),
    }
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, CHECKPOINT_PATH)


def read_checkpoint() -> tuple[str | None, str | None] | None:
    import json, os
    if not os.path.isfile(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return data.get("profile_id"), data.get("workspace_id")
    except Exception:
        return None


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


class StartupSelection:
    """Return value from the startup dashboard."""

    def __init__(self, profile_id: str, workspace_id: str, switched: bool = False) -> None:
        self.profile_id = profile_id
        self.workspace_id = workspace_id
        self.switched = switched


class StartupCLI:
    """Unified startup dashboard: profile + workspace selection, detail views, and launch decision."""

    @staticmethod
    async def run_startup_dashboard(
        profiles: list[ConfigurationProfile],
        workspaces: list[MemoryWorkspace],
        current_active: Optional[ConfigurationProfile],
        active_workspace: Optional[MemoryWorkspace],
        config_store: Any,
    ) -> Optional[StartupSelection]:
        """Present an interactive startup menu.
        Returns a StartupSelection with the resolved profile_id and workspace_id,
        or None to keep current defaults."""
        import sys
        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
            if sys.platform != "win32" or not sys.stdout.isatty():
                try:
                    sys.stdout.reconfigure(encoding="utf-8")
                except Exception:
                    pass
        console = Console(legacy_windows=False)
        profile_id = current_active.profile_id if current_active else (profiles[0].profile_id if profiles else None)
        workspace_id = active_workspace.workspace_id if active_workspace else None

        while True:
            # Re-read from store each iteration to reflect any mutations
            profiles = config_store.list_profiles(limit=9999)
            workspaces = config_store.list_workspaces()
            active_profile = config_store.get_active_profile()
            active_ws = config_store.get_active_workspace()
            profile_id = active_profile.profile_id if active_profile else (profiles[0].profile_id if profiles else None)
            workspace_id = active_ws.workspace_id if active_ws else None

            # ── Dashboard header (profile + workspace as one unit) ──
            try:
                header = Table.grid(padding=(0, 2))
                header.add_column(style="bold", no_wrap=True)
                header.add_column(style="white")
                header.add_row("Profile:", f"[cyan]{active_profile.name if active_profile else '-'}[/]")
                if active_ws:
                    ws_detail = f"[cyan]{active_ws.name}[/] ({active_ws.trade_count} trades"
                    size_str = f"{active_ws.size_bytes / 1024:.0f} KB)" if active_ws.size_bytes > 0 else "-)"
                    ws_line = ws_detail + ", " + size_str
                    header.add_row("  └─ Workspace:", ws_line)
                else:
                    header.add_row("  └─ Workspace:", "[dim]not linked[/]")
                console.print()
                console.print(Panel.fit(header, title="Active Setup", title_align="left", border_style="cyan"))
            except Exception:
                console.print("[yellow]Active Setup overview unavailable[/]")

            # ── Main menu ──
            try:
                action = await questionary.select(
                    "What would you like to do?",
                    choices=[
                        "Launch engine with current setup",
                        "Switch profile ...",
                        "View profile details",
                        "View detailed memory health",
                        "Manage workspaces ...",
                        "Delete profile ...",
                        "Quit",
                    ],

                ).ask_async()
            except KeyboardInterrupt:
                action = "Quit"
            except Exception as e:
                logger.warning("Dashboard menu error, falling back to quit", error=str(e))
                action = "Quit"

            if action == "Launch engine with current setup":
                if profile_id is None:
                    console.print("[red]No profile selected. Cannot launch.[/]")
                    continue
                write_checkpoint(profile_id, workspace_id)
                logger.info("Starting session with profile=%s workspace=%s", profile_id, workspace_id)
                console.print()
                console.print(
                    Panel.fit(
                        Text.assemble(
                            ("Starting Trading Session\n", "bold green"),
                            (f"Profile: {active_profile.name if active_profile else profile_id}\n", "white"),
                            (f"Workspace: {active_ws.name if active_ws else workspace_id}", "dim"),
                        ),
                        border_style="green",
                    )
                )
                return StartupSelection(profile_id=profile_id, workspace_id=workspace_id or "")

            elif action == "Switch profile ...":
                picked = await StartupCLI._profile_picker(profiles, current_active, config_store, console)
                if picked is not None:
                    profile_id = picked
                    config_store.activate_profile(profile_id, activated_by="operator")
                    picked_name = next((p.name for p in profiles if p.profile_id == picked), picked)
                    ws = config_store.ensure_profile_workspace(profile_id, picked_name)
                    config_store.switch_workspace(ws.workspace_id)
                    workspace_id = ws.workspace_id
                    write_checkpoint(profile_id, workspace_id)
                    console.print(f"[green]Profile switched to: {picked_name}[/]")
                    console.print(f"[dim]Workspace switched to: {ws.name}[/]")
                    logger.info("Profile switched", profile=picked_name, workspace_id=ws.workspace_id, _force_log=True)
                continue

            elif action == "View profile details":
                pid = await StartupCLI._profile_picker(profiles, current_active, config_store, console, detail_only=True)
                if pid is None:
                    pid = profile_id
                p = next((x for x in profiles if x.profile_id == pid), None)
                if p:
                    StartupCLI._display_profile_detail(p, config_store, console)
                continue

            elif action == "View detailed memory health":
                wid = workspace_id
                picked_ws = await StartupCLI._workspace_picker(workspaces, active_ws, config_store, console, detail_only=True)
                if picked_ws is not None:
                    wid = picked_ws
                if wid:
                    w = next((x for x in workspaces if x.workspace_id == wid), None)
                    if w:
                        await StartupCLI._display_memory_health_for_workspace(w, console)
                else:
                    console.print("[yellow]No workspace selected.[/]")
                continue

            elif action == "Manage workspaces ...":
                await StartupCLI.manage_workspaces(config_store)
                continue

            elif action == "Delete profile ...":
                picked_ids = await StartupCLI._profile_picker_multi(profiles, console)
                if picked_ids:
                    names = [next((x.name for x in profiles if x.profile_id == pid), pid) for pid in picked_ids]
                    confirm = await questionary.confirm(f"Delete {len(picked_ids)} profile(s) and their linked workspaces?\n  " + "\n  ".join(names), default=False).ask_async()
                    if confirm:
                        for pid in picked_ids:
                            config_store.delete_profile(pid, remove_workspace=True)
                        console.print(f"[red]{len(picked_ids)} profile(s) deleted.[/]")
                        logger.info("Profiles bulk deleted", count=len(picked_ids), profile_ids=picked_ids, _force_log=True)
                continue

            elif action is None or action == "Quit":
                logger.info("Startup dashboard quit by user")
                return None

        return None

    # ── Sub-helper: profile picker ──

    @staticmethod
    async def _profile_picker(
        profiles: list[ConfigurationProfile],
        current_active: Optional[ConfigurationProfile],
        config_store: Any,
        console: Console,
        detail_only: bool = False,
    ) -> Optional[str]:
        """Let the user pick a profile. Returns profile_id or None."""
        if not profiles:
            console.print("[yellow]No profiles available.[/]")
            return None

        choices = []
        for p in profiles:
            label = p.name
            if p.is_active:
                label += " (active)"
            choices.append(questionary.Choice(title=label, value=p.profile_id))

        if detail_only:
            prompt = "Select a profile to view"
        else:
            prompt = "Select a profile"

        result = await questionary.select(
            prompt,
            choices=choices,
        ).ask_async()

        return result

    @staticmethod
    async def _profile_picker_multi(
        profiles: list[ConfigurationProfile],
        console: Console,
    ) -> list[str]:
        """Let the user pick multiple profiles. Returns list of profile_ids or empty list."""
        if not profiles:
            console.print("[yellow]No profiles available.[/]")
            return []

        choices = []
        for p in profiles:
            label = p.name
            if p.is_active:
                label += " (active)"
            choices.append(questionary.Choice(title=label, value=p.profile_id))

        result = await questionary.checkbox(
            "Select profiles to delete (space to toggle, enter to confirm)",
            choices=choices,
        ).ask_async()

        return result or []

    # ── Sub-helper: workspace picker ──

    @staticmethod
    async def _workspace_picker(
        workspaces: list[MemoryWorkspace],
        active_ws: Optional[MemoryWorkspace],
        config_store: Any,
        console: Console,
        detail_only: bool = False,
    ) -> Optional[str]:
        """Let the user pick a workspace. Returns workspace_id or None."""
        if not workspaces:
            console.print("[yellow]No workspaces available.[/]")
            return None

        choices = []
        for w in workspaces:
            profile_tag = ""
            if w.profile_id:
                p = config_store.get_profile(w.profile_id)
                if p:
                    profile_tag = f" [{p.name}]"
            label = f"{w.name}{profile_tag}  ({w.trade_count} trades, {w.size_bytes // 1024} KB)"
            if w.is_active:
                label += " (active)"
            choices.append(questionary.Choice(title=label, value=w.workspace_id))

        if detail_only:
            prompt = "Select a workspace to view"
        else:
            prompt = "Select a workspace"

        result = await questionary.select(
            prompt,
            choices=choices,
        ).ask_async()

        return result

    # ── Profile detail view ──

    @staticmethod
    def _display_profile_detail(
        profile: ConfigurationProfile,
        config_store: Any,
        console: Console,
    ) -> None:
        """Show full profile configuration and active adaptations."""
        lines: list[tuple[str, str]] = [
            ("Profile ID", profile.profile_id),
            ("Name", profile.name),
            ("Type", "System" if profile.system_generated else "User"),
            ("Created", profile.created_at.strftime("%Y-%m-%d %H:%M UTC") if profile.created_at else "—"),
            ("Status", "Active" if profile.is_active else "Inactive"),
        ]
        if profile.workspace_id:
            ws = config_store.get_workspace(profile.workspace_id)
            if ws:
                lines.append(("Workspace", ws.name))
                lines.append(("WS Trades", str(ws.trade_count)))
                size_str = f"{ws.size_bytes / 1024:.0f} KB" if ws.size_bytes > 0 else "—"
                lines.append(("WS Size", size_str))
                lines.append(("WS Path", ws.db_path))
            else:
                lines.append(("Workspace", "[yellow]missing[/]"))
        else:
            lines.append(("Workspace", "[dim]not linked[/]"))

        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold dim", no_wrap=True)
        info.add_column(style="white")
        for label, value in lines:
            info.add_row(f"{label}:", value)

        console.print()
        console.print(Panel.fit(info, title=f"Profile Details: {profile.name}", title_align="left", border_style="cyan"))

        # ── Resolved configuration ──
        if profile.resolved_configuration:
            config_table = Table(
                title="Resolved Configuration",
                title_style="bold cyan",
                border_style="dim",
                box=None,
            )
            config_table.add_column("Key", style="bold", no_wrap=True)
            config_table.add_column("Value", style="white")
            for key in sorted(profile.resolved_configuration.keys()):
                val = profile.resolved_configuration[key]
                config_table.add_row(key, str(val) if val is not None else "—")
            console.print()
            console.print(config_table)
            console.print()

        # ── Active adaptive versions ──
        try:
            adaptations = config_store.get_active_adaptive_versions(profile.profile_id)
        except Exception:
            adaptations = None
        if adaptations:
            adapt_table = Table(
                title="Active Adaptive Parameters",
                title_style="bold yellow",
                border_style="dim",
                box=None,
            )
            adapt_table.add_column("Parameter", style="bold", no_wrap=True)
            adapt_table.add_column("Value", style="white")
            adapt_table.add_column("Version", style="dim")
            for param, val, ver in adaptations:
                adapt_table.add_row(param, str(val) if val is not None else "—", str(ver) if ver else "—")
            console.print()
            console.print(adapt_table)
            console.print()
        else:
            console.print("[dim]No active adaptive parameters.[/]")

        console.print()

    # ── Workspace detail view ──

    @staticmethod
    async def _display_workspace_detail(
        workspace: MemoryWorkspace,
        config_store: Any,
        console: Console,
    ) -> None:
        """Show workspace metadata and optionally memory health from corpus."""
        profile_name = "—"
        if workspace.profile_id:
            p = config_store.get_profile(workspace.profile_id)
            if p:
                profile_name = p.name
        lines: list[tuple[str, str]] = [
            ("Workspace ID", workspace.workspace_id),
            ("Name", workspace.name),
            ("Linked Profile", profile_name),
            ("DB Path", workspace.db_path or "—"),
            ("Trades", str(workspace.trade_count)),
            ("Size", f"{workspace.size_bytes / 1024:.1f} KB" if workspace.size_bytes > 0 else "—"),
            ("Last Used", workspace.last_used_at.strftime("%Y-%m-%d %H:%M UTC") if workspace.last_used_at else "—"),
            ("Status", "Active" if workspace.is_active else "Inactive"),
        ]

        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold dim", no_wrap=True)
        info.add_column(style="white")
        for label, value in lines:
            info.add_row(f"{label}:", value)

        console.print()
        console.print(Panel.fit(info, title=f"Workspace Details: {workspace.name}", title_align="left", border_style="cyan"))

        # ── Memory health from corpus if DB exists ──
        if workspace.db_path:
            await StartupCLI._display_memory_health_for_workspace(workspace, console)

        console.print()

    # ── Memory health for a workspace ──

    @staticmethod
    async def _display_memory_health_for_workspace(
        workspace: MemoryWorkspace,
        console: Console,
    ) -> None:
        """Open a temporary LearningCorpus on the workspace DB and show memory health."""
        if not workspace.db_path:
            console.print("[yellow]No database path for this workspace.[/]")
            return
        try:
            import os
            if not os.path.isfile(workspace.db_path):
                console.print("[yellow]Workspace database file does not exist yet. Start a session to initialize it.[/]")
                return
            corpus = LearningCorpus(db_path=workspace.db_path)
            StartupCLI.display_memory_health(corpus)
            del corpus
        except Exception as e:
            console.print(f"[red]Failed to read memory health: {e}[/]")

    @staticmethod
    async def run_memory_review(
        workspaces: list[MemoryWorkspace],
    ) -> Optional[str]:
        """Prompt operator to select an active memory workspace.
        Returns the workspace_id to activate, or None to keep current."""
        console = Console()

        if not workspaces:
            console.print("[yellow]No memory workspaces found. A default will be created on first use.[/]")
            return None

        ws_table = Table(title="Memory Workspaces", title_style="bold cyan", border_style="dim")
        ws_table.add_column("#", style="bold", width=3)
        ws_table.add_column("Name", style="white", no_wrap=True)
        ws_table.add_column("Trades", style="dim", justify="right")
        ws_table.add_column("Size", style="dim", justify="right")
        ws_table.add_column("Last Used", style="dim")
        ws_table.add_column("Status", style="dim", width=10)

        active_idx = 0
        for i, ws in enumerate(workspaces):
            size_str = f"{ws.size_bytes / 1024:.0f} KB" if ws.size_bytes > 0 else "—"
            last_used = ws.last_used_at.strftime("%Y-%m-%d") if ws.last_used_at else "—"
            status = "[green]Active[/]" if ws.is_active else ""
            if ws.is_active:
                active_idx = i
            ws_table.add_row(str(i + 1), ws.name, str(ws.trade_count), size_str, last_used, status)

        console.print()
        console.print(ws_table)
        console.print()

        try:
            choice = await asyncio.wait_for(
                asyncio.to_thread(input, f"Select workspace [{active_idx + 1}]: "),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, EOFError):
            logger.warning("Workspace selection timed out, keeping current")
            return None

        choice = choice.strip()
        if choice == "":
            return None

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(workspaces):
                console.print("[yellow]Invalid selection. Keeping current workspace.[/]")
                return None
        except ValueError:
            console.print("[yellow]Invalid input. Keeping current workspace.[/]")
            return None

        chosen = workspaces[idx]
        if chosen.is_active:
            return None

        console.print()
        console.print(
            Panel.fit(
                Text.assemble(
                    (f"Switching to workspace: {chosen.name}\n", "bold green"),
                    (f"DB: {chosen.db_path}", "dim"),
                ),
                border_style="green",
            )
        )

        return chosen.workspace_id

    @staticmethod
    def display_memory_health(corpus: LearningCorpus) -> None:
        console = Console()
        try:
            health = corpus.get_memory_health()
            table = Table(title="Memory Health", title_style="bold cyan", border_style="dim")
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="white")
            table.add_row("Experience Count", str(health.experience_count))
            table.add_row("Pending Candidates", str(health.pending_candidates))
            table.add_row("Rejected Candidates", str(health.rejected_count))
            table.add_row("Duplicates Merged", str(health.duplicate_count))
            table.add_row("Workspace Size", f"{health.workspace_size_bytes / 1024:.1f} KB")
            table.add_row("Database Size", f"{health.database_size_bytes / 1024:.1f} KB")
            table.add_row("Integrity State", f"[{'green' if health.integrity_state == 'ok' else 'red'}]{health.integrity_state}[/]")
            table.add_row("Verification State", f"[{'green' if health.verification_state == 'verified' else 'yellow'}]{health.verification_state}[/]")
            table.add_row("Last Maintenance", health.last_maintenance or "—")
            table.add_row("Last Save", health.last_save or "—")
            console.print()
            console.print(table)
            console.print()
        except Exception as e:
            console.print(f"[red]Failed to get memory health: {e}[/]")

    @staticmethod
    def display_pending_candidates(corpus: LearningCorpus) -> None:
        console = Console()
        try:
            candidates = corpus.get_pending_candidates()
            if not candidates:
                console.print("[yellow]No pending candidates.[/]")
                return
            table = Table(title=f"Pending Candidates ({len(candidates)})", title_style="bold cyan", border_style="dim")
            table.add_column("ID", style="dim", width=12)
            table.add_column("Position ID", style="dim", width=12)
            table.add_column("Status", style="white")
            table.add_column("Evidence", justify="right")
            table.add_column("Created", style="dim")
            for c in candidates[:20]:
                cid = c.get("candidate_id", "")[:12]
                pid = c.get("position_id", "")[:12]
                status = c.get("status", "?")
                ev = str(c.get("evidence_count", 0))
                created = str(c.get("created_at", ""))[:19] if c.get("created_at") else "—"
                table.add_row(cid, pid, status, ev, created)
            console.print()
            console.print(table)
            if len(candidates) > 20:
                console.print(f"[dim]... and {len(candidates) - 20} more[/]")
            console.print()
        except Exception as e:
            console.print(f"[red]Failed to list candidates: {e}[/]")

    @staticmethod
    def display_rejected_candidates(corpus: LearningCorpus) -> None:
        console = Console()
        try:
            rejected = corpus.get_rejected_candidates()
            if not rejected:
                console.print("[yellow]No rejected candidates.[/]")
                return
            table = Table(title=f"Rejected Candidates ({len(rejected)})", title_style="bold yellow", border_style="dim")
            table.add_column("ID", style="dim", width=12)
            table.add_column("Position", style="dim", width=12)
            table.add_column("Reason", style="white")
            table.add_column("Stage", style="dim")
            table.add_column("Created", style="dim")
            for r in rejected[:20]:
                rid = r.get("candidate_id", "")[:12]
                pid = r.get("position_id", "")[:12]
                reason = r.get("reject_reason", "?")
                stage = r.get("reject_stage", "?")
                created = str(r.get("created_at", ""))[:19] if r.get("created_at") else "—"
                table.add_row(rid, pid, reason, stage, created)
            console.print()
            console.print(table)
            if len(rejected) > 20:
                console.print(f"[dim]... and {len(rejected) - 20} more[/]")
            console.print()
        except Exception as e:
            console.print(f"[red]Failed to list rejected candidates: {e}[/]")

    @staticmethod
    def display_adaptive_history(config_store) -> None:
        console = Console()
        try:
            decisions = config_store.list_adaptive_decisions(limit=30)
            if not decisions:
                console.print("[yellow]No adaptive decisions yet.[/]")
                return
            table = Table(title="Adaptive Decision History", title_style="bold cyan", border_style="dim")
            table.add_column("Parameter", style="bold", no_wrap=True)
            table.add_column("Old Value", style="dim")
            table.add_column("→", width=2)
            table.add_column("New Value", style="bold")
            table.add_column("Confidence", justify="right")
            table.add_column("Evidence", justify="right")
            table.add_column("Status", style="white")
            table.add_column("Created", style="dim")
            for d in decisions:
                table.add_row(
                    d.parameter_id[:20],
                    str(d.current_value) if d.current_value is not None else "—",
                    "→",
                    str(d.proposed_value) if d.proposed_value is not None else "—",
                    f"{d.confidence:.2f}",
                    str(d.sample_count),
                    d.status,
                    d.created_at.strftime("%m-%d %H:%M") if d.created_at else "—",
                )
            console.print()
            console.print(table)
            console.print()
        except Exception as e:
            console.print(f"[red]Failed to list adaptive history: {e}[/]")

    @staticmethod
    async def select_learning_policy(config_store) -> Optional[str]:
        console = Console()
        policies = config_store.list_learning_policies()
        if not policies:
            console.print("[yellow]No learning policies found.[/]")
            return None

        table = Table(title="Learning Policies", title_style="bold cyan", border_style="dim")
        table.add_column("#", style="bold", width=3)
        table.add_column("Name", style="white", no_wrap=True)
        table.add_column("Tier", style="dim")
        table.add_column("Min Score", justify="right")
        table.add_column("Min Evidence", justify="right")
        table.add_column("Min Confidence", justify="right")
        table.add_column("Auto Approve", style="dim")
        table.add_column("Maint (h)", justify="right")
        table.add_column("Status", style="dim", width=10)

        active_idx = 0
        for i, p in enumerate(policies):
            status = "[green]Active[/]" if p.is_active else ""
            if p.is_active:
                active_idx = i
            table.add_row(
                str(i + 1), p.name, p.tier,
                str(p.validation_min_score),
                str(p.evidence_min_count),
                f"{p.confidence_min:.1f}",
                "[green]yes[/]" if p.auto_approve_candidates else "[yellow]no[/]",
                str(p.maintenance_interval_hours),
                status,
            )

        console.print()
        console.print(table)
        console.print()

        try:
            choice = await asyncio.wait_for(
                asyncio.to_thread(input, f"Select policy [{active_idx + 1}]: "),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, EOFError):
            logger.warning("Policy selection timed out, keeping current")
            return None

        choice = choice.strip()
        if choice == "":
            return None
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(policies):
                console.print("[yellow]Invalid selection. Keeping current policy.[/]")
                return None
        except ValueError:
            console.print("[yellow]Invalid input. Keeping current policy.[/]")
            return None

        chosen = policies[idx]
        if chosen.is_active:
            return None

        console.print()
        console.print(
            Panel.fit(
                Text.assemble(
                    (f"Switching to learning policy: {chosen.name}\n", "bold green"),
                    (f"Tier: {chosen.tier}", "dim"),
                ),
                border_style="green",
            )
        )
        return chosen.policy_id

    @staticmethod
    def display_confidence_evolution(corpus: LearningCorpus) -> None:
        console = Console()
        try:
            health = corpus.get_memory_health()
            table = Table(title="Confidence Evolution", title_style="bold cyan", border_style="dim")
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="white")
            table.add_row("Experience Count", str(health.experience_count))
            table.add_row("Pending Candidates", str(health.pending_candidates))
            table.add_row("Rejected Count", str(health.rejected_count))
            table.add_row("Duplicates Merged", str(health.duplicate_count))
            table.add_row("Integrity State", health.integrity_state)
            console.print()
            console.print(table)
            console.print()
        except Exception as e:
            console.print(f"[red]Failed to get confidence evolution: {e}[/]")

    @staticmethod
    async def manage_workspaces(config_store) -> Optional[str]:
        console = Console()
        workspaces = config_store.list_workspaces()
        if not workspaces:
            console.print("[yellow]No workspaces found.[/]")
            return None

        table = Table(title="Manage Workspaces", title_style="bold cyan", border_style="dim")
        table.add_column("#", style="bold", width=3)
        table.add_column("Name", style="white", no_wrap=True)
        table.add_column("Profile", style="cyan", no_wrap=True)
        table.add_column("Trades", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Active", style="dim")
        for i, ws in enumerate(workspaces):
            profile_tag = ""
            if ws.profile_id:
                p = config_store.get_profile(ws.profile_id)
                if p:
                    profile_tag = p.name
            size_str = f"{ws.size_bytes / 1024:.0f} KB" if ws.size_bytes > 0 else "—"
            status = "[green]Active[/]" if ws.is_active else ""
            table.add_row(str(i + 1), ws.name, profile_tag, str(ws.trade_count), size_str, status)

        console.print()
        console.print(table)
        console.print()

        console.print("[dim]Actions: switch (number) | archive (a+number) | export (e+number) | clear (c+number)[/]")
        try:
            cmd = await asyncio.wait_for(
                asyncio.to_thread(input, "Command: "),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, EOFError):
            logger.warning("Workspace management timed out")
            return None

        cmd = cmd.strip().lower()
        if not cmd:
            return None

        try:
            if cmd.startswith("a"):
                idx = int(cmd[1:]) - 1
                if 0 <= idx < len(workspaces):
                    config_store.archive_workspace(workspaces[idx].workspace_id)
                    console.print(f"[green]Workspace archived: {workspaces[idx].name}[/]")
                    return None
            elif cmd.startswith("e"):
                idx = int(cmd[1:]) - 1
                if 0 <= idx < len(workspaces):
                    export_path = f"data/export_{workspaces[idx].workspace_id[:8]}.duckdb"
                    config_store.export_workspace(workspaces[idx].workspace_id, export_path)
                    console.print(f"[green]Workspace exported to: {export_path}[/]")
                    return None
            elif cmd.startswith("c"):
                idx = int(cmd[1:]) - 1
                if 0 <= idx < len(workspaces):
                    count = config_store.clear_workspace(workspaces[idx].workspace_id)
                    console.print(f"[yellow]Workspace cleared: {count} tables emptied[/]")
                    return None
            else:
                idx = int(cmd) - 1
                if 0 <= idx < len(workspaces):
                    chosen = workspaces[idx]
                    if not chosen.is_active:
                        config_store.switch_workspace(chosen.workspace_id)
                        console.print(f"[green]Switched to workspace: {chosen.name}[/]")
                    return chosen.workspace_id
        except (ValueError, IndexError):
            console.print("[red]Invalid command.[/]")
            return None
        return None
