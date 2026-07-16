from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import duckdb
import structlog

from src.recommendations.models import (
    ActivationRecord,
    AdaptiveDecision,
    AdaptiveParameter,
    AdaptiveVersion,
    ConfigurationProfile,
    Finding,
    Intervention,
    LearningPolicy,
    MemoryWorkspace,
    Recommendation,
)

logger = structlog.get_logger("config_store")

CONFIG_DB = "data/configuration_profiles.duckdb"


class ConfigurationStore:
    """Append-only DuckDB storage for recommendations, profiles, and activation history."""

    def __init__(self, db_path: str = CONFIG_DB):
        self._conn = duckdb.connect(db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                finding_id VARCHAR PRIMARY KEY,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                intervention_id VARCHAR PRIMARY KEY,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rec_status
            ON recommendations (status)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id VARCHAR PRIMARY KEY,
                is_active BOOLEAN NOT NULL DEFAULT FALSE,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            DROP INDEX IF EXISTS idx_profiles_active
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS activation_history (
                record_id VARCHAR PRIMARY KEY,
                profile_id VARCHAR NOT NULL,
                activated_at TIMESTAMP NOT NULL,
                deactivated_at TIMESTAMP,
                activated_by VARCHAR NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_activation_profile
            ON activation_history (profile_id)
        """)

        # ── Adaptive parameters (Category B) ──
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_parameters (
                parameter_id VARCHAR PRIMARY KEY,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_versions (
                version_id VARCHAR PRIMARY KEY,
                parameter_id VARCHAR NOT NULL,
                profile_id VARCHAR NOT NULL,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_adaptive_versions_param
            ON adaptive_versions (parameter_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_adaptive_versions_profile
            ON adaptive_versions (profile_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_decisions (
                decision_id VARCHAR PRIMARY KEY,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS version_tracking (
                position_id VARCHAR PRIMARY KEY,
                profile_id VARCHAR NOT NULL,
                version_snapshot JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_version_tracking_profile
            ON version_tracking (profile_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS version_effectiveness (
                id INTEGER PRIMARY KEY,
                version_id VARCHAR NOT NULL,
                position_id VARCHAR NOT NULL,
                parameter_id VARCHAR NOT NULL,
                realized_pnl DOUBLE NOT NULL,
                effective_confidence DOUBLE NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_version_effectiveness_ver
            ON version_effectiveness (version_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_workspaces (
                workspace_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                db_path VARCHAR NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT FALSE,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            DROP INDEX IF EXISTS idx_memory_active
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_policies (
                policy_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                tier VARCHAR NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                payload_json JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_feedback_state (
                setting_name VARCHAR PRIMARY KEY,
                value VARCHAR NOT NULL,
                reason VARCHAR NOT NULL DEFAULT '',
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._seed_adaptive_parameters()
        self._seed_learning_policies()

    # ── Adaptive Parameters (definitions) ──

    ADAPTIVE_PARAMETER_DEFAULTS: list[dict[str, Any]] = [
        {
            "parameter_id": "trailing_atr_multiplier",
            "config_path": "execution.trailing_stop_atr_mult",
            "display_name": "Trailing Stop ATR Multiplier",
            "description": "Multiplier applied to ATR for trailing stop distance",
            "default_value": 2.0, "min_value": 1.0, "max_value": 5.0, "step": 0.1,
            "unit": "multiplier", "decay_rate": 0.005, "required_evidence_count": 20,
        },
        {
            "parameter_id": "stop_loss_pct",
            "config_path": "execution.stop_loss_pct",
            "display_name": "Stop Loss Percentage",
            "description": "Fraction of entry price for initial stop loss (LONG: below entry)",
            "default_value": 0.98, "min_value": 0.90, "max_value": 0.99, "step": 0.005,
            "unit": "pct", "decay_rate": 0.003, "required_evidence_count": 20,
        },
        {
            "parameter_id": "take_profit_pct",
            "config_path": "execution.take_profit_pct",
            "display_name": "Take Profit Percentage",
            "description": "Fraction of entry price for initial take profit (LONG: above entry)",
            "default_value": 1.04, "min_value": 1.01, "max_value": 1.15, "step": 0.005,
            "unit": "pct", "decay_rate": 0.003, "required_evidence_count": 20,
        },
        {
            "parameter_id": "slippage_bps",
            "config_path": "execution.slippage_bps",
            "display_name": "Slippage Estimate",
            "description": "Expected slippage in basis points for virtual execution simulation",
            "default_value": 3.0, "min_value": 0.5, "max_value": 20.0, "step": 0.5,
            "unit": "bps", "decay_rate": 0.002, "required_evidence_count": 15,
        },
        {
            "parameter_id": "fee_bps",
            "config_path": "execution.fee_bps",
            "display_name": "Fee Estimate",
            "description": "Expected taker fee in basis points for virtual execution simulation",
            "default_value": 4.0, "min_value": 1.0, "max_value": 10.0, "step": 0.5,
            "unit": "bps", "decay_rate": 0.001, "required_evidence_count": 10,
        },
        {
            "parameter_id": "spread_bps",
            "config_path": "execution.spread_bps",
            "display_name": "Spread Estimate",
            "description": "Expected bid-ask spread in basis points for virtual execution simulation",
            "default_value": 2.0, "min_value": 0.5, "max_value": 10.0, "step": 0.5,
            "unit": "bps", "decay_rate": 0.002, "required_evidence_count": 15,
        },
        {
            "parameter_id": "min_llm_confidence",
            "config_path": "risk.min_llm_confidence",
            "display_name": "Minimum LLM Confidence",
            "description": "Minimum LLM confidence threshold for trade execution",
            "default_value": 0.3, "min_value": 0.1, "max_value": 0.8, "step": 0.05,
            "unit": "score", "decay_rate": 0.005, "required_evidence_count": 30,
        },
        {
            "parameter_id": "evidence_tier_exact_threshold",
            "config_path": "evidence.exact_threshold",
            "display_name": "EXACT Evidence Tier Similarity Threshold",
            "description": "Minimum similarity to use EXACT evidence tier",
            "default_value": 0.0, "min_value": 0.0, "max_value": 0.6, "step": 0.05,
            "unit": "score", "decay_rate": 0.003, "required_evidence_count": 20,
        },
        {
            "parameter_id": "protection_audit_interval_seconds",
            "config_path": "execution.protection_audit_interval",
            "display_name": "Protection Audit Interval",
            "description": "Seconds between protection order audits on the exchange",
            "default_value": 60.0, "min_value": 30.0, "max_value": 300.0, "step": 10.0,
            "unit": "seconds", "decay_rate": 0.001, "required_evidence_count": 10,
        },
    ]

    def _seed_adaptive_parameters(self) -> None:
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM adaptive_parameters"
        ).fetchone()[0]
        if existing > 0:
            return
        for param in self.ADAPTIVE_PARAMETER_DEFAULTS:
            model = AdaptiveParameter(**param)
            self._conn.execute(
                """
                INSERT INTO adaptive_parameters (parameter_id, payload_json, created_at)
                VALUES (?, ?, ?)
                """,
                [model.parameter_id, json.dumps(model.model_dump(mode="json")), datetime.utcnow()],
            )
        logger.info(
            "Adaptive parameters seeded",
            count=len(self.ADAPTIVE_PARAMETER_DEFAULTS),
        )

    def get_all_adaptive_parameters(self) -> dict[str, AdaptiveParameter]:
        rows = self._conn.execute(
            "SELECT payload_json FROM adaptive_parameters ORDER BY created_at"
        ).fetchall()
        return {
            data["parameter_id"]: AdaptiveParameter(**data)
            for r in rows
            if (data := json.loads(r[0]))
        }

    # ── Adaptive Versions (parameter values) ──

    def get_active_adaptive_versions(self, profile_id: str) -> dict[str, AdaptiveVersion]:
        rows = self._conn.execute(
            "SELECT payload_json FROM adaptive_versions "
            "WHERE profile_id = ? ORDER BY created_at DESC",
            [profile_id],
        ).fetchall()
        versions: dict[str, AdaptiveVersion] = {}
        for row in rows:
            v = AdaptiveVersion.model_validate(json.loads(row[0]))
            if v.status == "active" and v.parameter_id not in versions:
                versions[v.parameter_id] = v
        return versions

    def save_adaptive_version(self, version: AdaptiveVersion) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO adaptive_versions
                (version_id, parameter_id, profile_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                version.version_id,
                version.parameter_id,
                version.profile_id,
                json.dumps(version.model_dump(mode="json")),
                version.created_at.isoformat(),
            ],
        )
        logger.info(
            "AdaptiveVersion saved",
            parameter_id=version.parameter_id,
            value=version.value,
            status=version.status,
        )

    def supersede_adaptive_version(self, parameter_id: str, profile_id: str) -> None:
        active = self.get_active_adaptive_versions(profile_id).get(parameter_id)
        if active is None:
            return
        updated = active.model_dump(mode="json")
        updated["status"] = "superseded"
        updated["superseded_at"] = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE adaptive_versions SET payload_json = ? WHERE version_id = ?",
            [json.dumps(updated), active.version_id],
        )
        logger.info(
            "AdaptiveVersion superseded",
            parameter_id=parameter_id, version_id=active.version_id,
        )

    def get_adaptive_version_history(
        self, parameter_id: str, profile_id: str, limit: int = 20,
    ) -> list[AdaptiveVersion]:
        rows = self._conn.execute(
            "SELECT payload_json FROM adaptive_versions "
            "WHERE parameter_id = ? AND profile_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            [parameter_id, profile_id, limit],
        ).fetchall()
        return [AdaptiveVersion.model_validate(json.loads(r[0])) for r in rows]

    # ── Adaptive Decisions ──

    def save_adaptive_decision(self, decision: AdaptiveDecision) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO adaptive_decisions
                (decision_id, payload_json, created_at)
            VALUES (?, ?, ?)
            """,
            [
                decision.decision_id,
                json.dumps(decision.model_dump(mode="json")),
                decision.created_at.isoformat(),
            ],
        )
        logger.info(
            "AdaptiveDecision saved",
            decision_id=decision.decision_id,
            parameter_id=decision.parameter_id,
            status=decision.status,
        )

    def list_adaptive_decisions(
        self, status_filter: Optional[str] = None, limit: int = 20,
    ) -> list[AdaptiveDecision]:
        rows = self._conn.execute(
            "SELECT payload_json FROM adaptive_decisions ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        decisions = [AdaptiveDecision.model_validate(json.loads(r[0])) for r in rows]
        if status_filter:
            decisions = [d for d in decisions if d.status == status_filter]
        return decisions[:limit]

    def update_decision_status(
        self, decision_id: str, new_status: str, reason: str = "",
    ) -> None:
        rows = self._conn.execute(
            "SELECT payload_json FROM adaptive_decisions WHERE decision_id = ?",
            [decision_id],
        ).fetchall()
        if not rows:
            logger.warning("AdaptiveDecision not found", decision_id=decision_id)
            return
        data = json.loads(rows[0][0])
        data["status"] = new_status
        if reason:
            data["reason"] = reason
        if new_status in ("approved", "rejected", "applied", "superseded"):
            data["decided_at"] = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE adaptive_decisions SET payload_json = ? WHERE decision_id = ?",
            [json.dumps(data), decision_id],
        )
        logger.info(
            "AdaptiveDecision status updated",
            decision_id=decision_id, new_status=new_status, reason=reason,
        )

    # ── Version Tracking (which versions were active per trade) ──

    def record_version_snapshot(
        self, position_id: str, profile_id: str,
        active_versions: dict[str, AdaptiveVersion],
    ) -> None:
        snapshot = {k: v.version_id for k, v in active_versions.items()}
        self._conn.execute(
            """
            INSERT OR REPLACE INTO version_tracking
                (position_id, profile_id, version_snapshot, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [position_id, profile_id, json.dumps(snapshot), datetime.utcnow().isoformat()],
        )
        logger.debug("Version snapshot recorded", position_id=position_id, snapshot=snapshot)

    def get_version_snapshot(self, position_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT version_snapshot FROM version_tracking WHERE position_id = ?",
            [position_id],
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]) if isinstance(row[0], str) else row[0]

    # ── Version Effectiveness (outcome per version) ──

    def record_version_outcome(
        self, version_id: str, position_id: str, parameter_id: str,
        realized_pnl: float, effective_confidence: float,
    ) -> None:
        max_id = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM version_effectiveness").fetchone()[0]
        self._conn.execute(
            """
            INSERT INTO version_effectiveness
                (id, version_id, position_id, parameter_id, realized_pnl,
                 effective_confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [max_id + 1, version_id, position_id, parameter_id,
             realized_pnl, effective_confidence, datetime.utcnow().isoformat()],
        )

    def get_version_effectiveness(self, version_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM version_effectiveness WHERE version_id = ? ORDER BY created_at",
            [version_id],
        ).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, r)) for r in rows]

    def get_aggregate_effectiveness(self, parameter_id: str) -> dict:
        rows = self._conn.execute(
            """
            SELECT ve.version_id, ve.realized_pnl, ve.effective_confidence
            FROM version_effectiveness ve
            WHERE ve.parameter_id = ?
            """,
            [parameter_id],
        ).fetchall()
        if not rows:
            return {"parameter_id": parameter_id, "samples": 0}

        by_version: dict[str, list[float]] = {}
        for version_id, pnl, conf in rows:
            by_version.setdefault(version_id, []).append(pnl)

        best_version = max(by_version, key=lambda v: sum(by_version[v]))
        worst_version = min(by_version, key=lambda v: sum(by_version[v]))

        return {
            "parameter_id": parameter_id,
            "samples": len(rows),
            "versions_used": len(by_version),
            "best_version_id": best_version,
            "best_avg_pnl": round(sum(by_version[best_version]) / len(by_version[best_version]), 2),
            "worst_version_id": worst_version,
            "worst_avg_pnl": round(sum(by_version[worst_version]) / len(by_version[worst_version]), 2),
            "overall_avg_pnl": round(sum(p for _, p, _ in rows) / len(rows), 2),
        }

    # ── Memory Workspaces ──

    def list_workspaces(self) -> list[MemoryWorkspace]:
        rows = self._conn.execute(
            "SELECT payload_json FROM memory_workspaces ORDER BY created_at DESC",
        ).fetchall()
        return [MemoryWorkspace.model_validate(json.loads(r[0])) for r in rows]

    def ensure_default_workspace(self) -> MemoryWorkspace:
        """Ensure an active memory workspace exists and points at the default corpus DB."""
        import os
        import uuid

        active = self.get_active_workspace()
        if active is not None:
            return active

        existing = self.list_workspaces()
        if existing:
            self.switch_workspace(existing[0].workspace_id)
            return existing[0]

        default_path = "data/experience_corpus.duckdb"
        os.makedirs(os.path.dirname(default_path) or ".", exist_ok=True)
        ws = MemoryWorkspace(
            workspace_id=str(uuid.uuid4()),
            name="default",
            db_path=default_path,
            is_active=True,
            description="Default learning memory workspace",
        )
        self.save_workspace(ws)
        logger.info("Default memory workspace created", db_path=default_path)
        return ws

    def save_workspace(self, workspace: MemoryWorkspace) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memory_workspaces
                (workspace_id, name, db_path, is_active, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                workspace.workspace_id,
                workspace.name,
                workspace.db_path,
                workspace.is_active,
                json.dumps(workspace.model_dump(mode="json")),
                workspace.created_at.isoformat(),
            ],
        )
        logger.info("MemoryWorkspace saved", name=workspace.name, is_active=workspace.is_active)

    def get_active_workspace(self) -> Optional[MemoryWorkspace]:
        row = self._conn.execute(
            "SELECT payload_json FROM memory_workspaces WHERE is_active LIMIT 1",
        ).fetchone()
        if row is None:
            return None
        return MemoryWorkspace.model_validate(json.loads(row[0]))

    def switch_workspace(self, workspace_id: str) -> None:
        self._conn.execute("UPDATE memory_workspaces SET is_active = FALSE WHERE is_active")
        self._conn.execute(
            "UPDATE memory_workspaces SET is_active = TRUE WHERE workspace_id = ?",
            [workspace_id],
        )
        logger.info("Memory workspace switched", workspace_id=workspace_id)

    def delete_workspace(self, workspace_id: str, remove_db_file: bool = True, unlink_profile: bool = True) -> None:
        ws = self.get_workspace(workspace_id)
        if unlink_profile and ws and ws.profile_id:
            profile = self.get_profile(ws.profile_id)
            if profile:
                updated = ConfigurationProfile(
                    profile_id=profile.profile_id,
                    name=profile.name,
                    created_at=profile.created_at,
                    base_profile=profile.base_profile,
                    parent_profile=profile.parent_profile,
                    description=profile.description,
                    tags=list(profile.tags),
                    notes=profile.notes,
                    system_generated=profile.system_generated,
                    parameter_overrides=dict(profile.parameter_overrides),
                    resolved_configuration=dict(profile.resolved_configuration),
                    is_active=profile.is_active,
                    workspace_id=None,
                    derived_from_recommendations=list(profile.derived_from_recommendations),
                    derived_from_findings=list(profile.derived_from_findings),
                    activation_reason=profile.activation_reason,
                    created_by=profile.created_by,
                )
                self.save_profile(updated)
                logger.info("Unlinked profile from deleted workspace", profile_id=ws.profile_id, workspace_id=workspace_id)
        self._conn.execute("DELETE FROM memory_workspaces WHERE workspace_id = ?", [workspace_id])
        if ws and ws.db_path and remove_db_file:
            import os
            try:
                if os.path.exists(ws.db_path):
                    os.remove(ws.db_path)
            except Exception as e:
                logger.warning("Failed to remove workspace DB file", db_path=ws.db_path, error=str(e))
        logger.info("Memory workspace deleted", workspace_id=workspace_id)

    def delete_profile(self, profile_id: str, remove_workspace: bool = True) -> None:
        """Delete a profile and optionally its linked workspace + DB file."""
        profile = self.get_profile(profile_id)
        if profile is None:
            return
        if remove_workspace and profile.workspace_id:
            self.delete_workspace(profile.workspace_id, remove_db_file=True)
        self._conn.execute("DELETE FROM activation_history WHERE profile_id = ?", [profile_id])
        self._conn.execute("DELETE FROM profiles WHERE profile_id = ?", [profile_id])
        logger.info("Profile deleted", profile_id=profile_id, name=profile.name, _force_log=True)

    def increment_workspace_trade_count(self, workspace_id: str) -> None:
        row = self._conn.execute(
            "SELECT payload_json FROM memory_workspaces WHERE workspace_id = ?",
            [workspace_id],
        ).fetchone()
        if row is None:
            logger.warning("Workspace not found for trade_count increment", workspace_id=workspace_id)
            return
        ws = MemoryWorkspace.model_validate(json.loads(row[0]))
        updated = MemoryWorkspace(
            workspace_id=ws.workspace_id,
            name=ws.name,
            db_path=ws.db_path,
            is_active=ws.is_active,
            description=ws.description,
            trade_count=ws.trade_count + 1,
            size_bytes=ws.size_bytes,
            created_at=ws.created_at,
            last_used_at=datetime.utcnow(),
        )
        self.save_workspace(updated)
        logger.info("Workspace trade_count incremented", workspace_id=workspace_id, trade_count=updated.trade_count)

    def get_workspace(self, workspace_id: str) -> Optional[MemoryWorkspace]:
        row = self._conn.execute(
            "SELECT payload_json FROM memory_workspaces WHERE workspace_id = ?",
            [workspace_id],
        ).fetchone()
        if row is None:
            return None
        return MemoryWorkspace.model_validate(json.loads(row[0]))

    def _link_profile_to_workspace(self, profile_id: str, workspace_id: str) -> None:
        profile = self.get_profile(profile_id)
        if profile is None:
            return
        updated = ConfigurationProfile(
            profile_id=profile.profile_id,
            name=profile.name,
            created_at=profile.created_at,
            base_profile=profile.base_profile,
            parent_profile=profile.parent_profile,
            description=profile.description,
            tags=list(profile.tags),
            notes=profile.notes,
            system_generated=profile.system_generated,
            parameter_overrides=dict(profile.parameter_overrides),
            resolved_configuration=dict(profile.resolved_configuration),
            is_active=profile.is_active,
            workspace_id=workspace_id,
            derived_from_recommendations=list(profile.derived_from_recommendations),
            derived_from_findings=list(profile.derived_from_findings),
            activation_reason=profile.activation_reason,
            created_by=profile.created_by,
        )
        self.save_profile(updated)
        # Also set profile_id on the workspace for bidirectional link
        ws = self.get_workspace(workspace_id)
        if ws and ws.profile_id != profile_id:
            updated_ws = MemoryWorkspace(
                workspace_id=ws.workspace_id,
                name=ws.name,
                db_path=ws.db_path,
                is_active=ws.is_active,
                description=ws.description,
                trade_count=ws.trade_count,
                size_bytes=ws.size_bytes,
                profile_id=profile_id,
                created_at=ws.created_at,
                last_used_at=ws.last_used_at,
            )
            self.save_workspace(updated_ws)

    def ensure_profile_workspace(self, profile_id: str, profile_name: str) -> MemoryWorkspace:
        """Ensure a workspace exists linked to the given profile. Returns existing or creates new."""
        import os
        import uuid

        profile = self.get_profile(profile_id)
        if profile and profile.workspace_id:
            existing = self.get_workspace(profile.workspace_id)
            if existing:
                return existing

        ws_name = f"{profile_name} workspace"
        existing = self.list_workspaces()
        for ws in existing:
            if ws.name == ws_name:
                self._link_profile_to_workspace(profile_id, ws.workspace_id)
                return ws

        ws_dir = "data/workspaces"
        os.makedirs(ws_dir, exist_ok=True)
        ws_path = f"{ws_dir}/{profile_id}.duckdb"
        ws = MemoryWorkspace(
            workspace_id=str(uuid.uuid4()),
            name=ws_name,
            db_path=ws_path,
            is_active=False,
            profile_id=profile_id,
            description=f"Memory workspace for profile: {profile_name}",
        )
        self.save_workspace(ws)
        self._link_profile_to_workspace(profile_id, ws.workspace_id)
        logger.info("Workspace created for profile", profile_id=profile_id, workspace_id=ws.workspace_id, name=ws_name, _force_log=True)
        return ws

    # ── Findings ──

    def save_finding(self, finding: Finding) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO findings (finding_id, payload_json, created_at)
            VALUES (?, ?, ?)
            """,
            [
                finding.finding_id,
                json.dumps(finding.model_dump(mode="json")),
                finding.created_at.isoformat(),
            ],
        )
        logger.debug("Finding saved", finding_id=finding.finding_id, category=finding.category)

    # ── Interventions ──

    def save_intervention(self, intervention: Intervention) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO interventions (intervention_id, payload_json, created_at)
            VALUES (?, ?, ?)
            """,
            [
                intervention.intervention_id,
                json.dumps(intervention.model_dump(mode="json")),
                intervention.created_at.isoformat(),
            ],
        )
        logger.debug(
            "Intervention saved",
            intervention_id=intervention.intervention_id,
            parameter_id=intervention.parameter_id,
        )

    # ── Recommendations ──

    def get_intervention(self, intervention_id: str) -> Optional[Intervention]:
        row = self._conn.execute(
            "SELECT payload_json FROM interventions WHERE intervention_id = ?",
            [intervention_id],
        ).fetchone()
        if row is None:
            return None
        return Intervention.model_validate(json.loads(row[0]))

    def save_recommendation(self, rec: Recommendation) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO recommendations
                (recommendation_id, status, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                rec.recommendation_id,
                rec.status,
                json.dumps(rec.model_dump(mode="json")),
                rec.created_at.isoformat(),
            ],
        )
        logger.info(
            "Recommendation saved",
            recommendation_id=rec.recommendation_id,
            status=rec.status,
            confidence_tier=rec.confidence_tier,
        )

    def list_recommendations(
        self, status_filter: Optional[str] = None, limit: int = 20,
    ) -> list[Recommendation]:
        if status_filter:
            rows = self._conn.execute(
                "SELECT payload_json FROM recommendations WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                [status_filter, limit],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM recommendations ORDER BY created_at DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [Recommendation.model_validate(json.loads(r[0])) for r in rows]

    # ── Profiles ──

    def save_profile(self, profile: ConfigurationProfile) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO profiles
                (profile_id, is_active, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                profile.profile_id,
                profile.is_active,
                json.dumps(profile.model_dump(mode="json")),
                profile.created_at.isoformat(),
            ],
        )
        logger.info("Profile saved", profile_id=profile.profile_id, name=profile.name)

    def get_active_profile(self) -> Optional[ConfigurationProfile]:
        row = self._conn.execute(
            "SELECT payload_json FROM profiles WHERE is_active LIMIT 1",
        ).fetchone()
        if row is None:
            return None
        return ConfigurationProfile.model_validate(json.loads(row[0]))

    def activate_profile(
        self,
        profile_id: str,
        activated_by: str = "system",
    ) -> None:
        self._conn.execute("BEGIN TRANSACTION")
        try:
            self._conn.execute(
                "UPDATE profiles SET is_active = FALSE WHERE is_active",
            )
            self._conn.execute(
                "UPDATE profiles SET is_active = TRUE WHERE profile_id = ?",
                [profile_id],
            )
            self._conn.execute(
                "UPDATE activation_history SET deactivated_at = ? WHERE profile_id = ? AND deactivated_at IS NULL",
                [datetime.utcnow().isoformat(), profile_id],
            )
            record = ActivationRecord(
                profile_id=profile_id,
                activated_by=activated_by,
            )
            self._conn.execute(
                """
                INSERT INTO activation_history
                    (record_id, profile_id, activated_at, deactivated_at, activated_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    record.record_id,
                    record.profile_id,
                    record.activated_at.isoformat(),
                    record.deactivated_at,
                    record.activated_by,
                ],
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        logger.info(
            "Profile activated",
            profile_id=profile_id,
            activated_by=activated_by,
        )

    def list_profiles(self, limit: int = 10) -> list[ConfigurationProfile]:
        rows = self._conn.execute(
            "SELECT payload_json FROM profiles ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [ConfigurationProfile.model_validate(json.loads(r[0])) for r in rows]

    def get_profile(self, profile_id: str) -> Optional[ConfigurationProfile]:
        row = self._conn.execute(
            "SELECT payload_json FROM profiles WHERE profile_id = ?",
            [profile_id],
        ).fetchone()
        if row is None:
            return None
        return ConfigurationProfile.model_validate(json.loads(row[0]))

    def get_activation_history(
        self, profile_id: Optional[str] = None, limit: int = 20,
    ) -> list[ActivationRecord]:
        if profile_id:
            rows = self._conn.execute(
                "SELECT * FROM activation_history WHERE profile_id = ? ORDER BY activated_at DESC LIMIT ?",
                [profile_id, limit],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM activation_history ORDER BY activated_at DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [
            ActivationRecord(
                record_id=r[0],
                profile_id=r[1],
                activated_at=datetime.fromisoformat(r[2]) if isinstance(r[2], str) else r[2],
                deactivated_at=datetime.fromisoformat(r[3]) if isinstance(r[3], str) and r[3] else None,
                activated_by=r[4],
            )
            for r in rows
        ]

    # ── Learning Policies ──

    LEARNING_POLICY_PRESETS: list[dict[str, Any]] = [
        {
            "policy_id": "policy_research",
            "name": "Research",
            "tier": "research",
            "validation_min_score": 90,
            "evidence_min_count": 30,
            "confidence_min": 0.8,
            "noise_max_score": 0.1,
            "auto_approve_candidates": False,
            "maintenance_interval_hours": 24,
            "confidence_decay_rate": 0.001,
            "duplicate_threshold": 0.85,
            "consolidation_threshold": 0.9,
            "verification_strictness": "strict",
        },
        {
            "policy_id": "policy_conservative",
            "name": "Conservative",
            "tier": "conservative",
            "validation_min_score": 80,
            "evidence_min_count": 20,
            "confidence_min": 0.6,
            "noise_max_score": 0.2,
            "auto_approve_candidates": False,
            "maintenance_interval_hours": 12,
            "confidence_decay_rate": 0.002,
            "duplicate_threshold": 0.85,
            "consolidation_threshold": 0.9,
            "verification_strictness": "strict",
        },
        {
            "policy_id": "policy_balanced",
            "name": "Balanced",
            "tier": "balanced",
            "is_active": True,
            "validation_min_score": 70,
            "evidence_min_count": 10,
            "confidence_min": 0.4,
            "noise_max_score": 0.3,
            "auto_approve_candidates": True,
            "maintenance_interval_hours": 6,
            "confidence_decay_rate": 0.005,
            "duplicate_threshold": 0.85,
            "consolidation_threshold": 0.9,
            "verification_strictness": "normal",
        },
        {
            "policy_id": "policy_aggressive",
            "name": "Aggressive",
            "tier": "aggressive",
            "validation_min_score": 60,
            "evidence_min_count": 5,
            "confidence_min": 0.2,
            "noise_max_score": 0.5,
            "auto_approve_candidates": True,
            "maintenance_interval_hours": 1,
            "confidence_decay_rate": 0.01,
            "duplicate_threshold": 0.8,
            "consolidation_threshold": 0.85,
            "verification_strictness": "relaxed",
        },
    ]

    def _seed_learning_policies(self) -> None:
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM learning_policies"
        ).fetchone()[0]
        if existing > 0:
            return
        for preset in self.LEARNING_POLICY_PRESETS:
            model = LearningPolicy(**preset)
            self._conn.execute("""
                INSERT INTO learning_policies (policy_id, name, tier, is_active, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                model.policy_id,
                model.name,
                model.tier,
                model.is_active,
                json.dumps(model.model_dump(mode="json")),
                datetime.utcnow(),
            ])
        logger.info("Learning policies seeded", count=len(self.LEARNING_POLICY_PRESETS))

    def save_learning_policy(self, policy: LearningPolicy) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO learning_policies
                (policy_id, name, tier, is_active, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            policy.policy_id,
            policy.name,
            policy.tier,
            policy.is_active,
            json.dumps(policy.model_dump(mode="json")),
            datetime.utcnow(),
        ])

    def get_active_learning_policy(self) -> Optional[LearningPolicy]:
        row = self._conn.execute(
            "SELECT payload_json FROM learning_policies WHERE is_active LIMIT 1",
        ).fetchone()
        if row is None:
            return None
        return LearningPolicy.model_validate(json.loads(row[0]))

    def list_learning_policies(self) -> list[LearningPolicy]:
        rows = self._conn.execute(
            "SELECT payload_json FROM learning_policies ORDER BY created_at",
        ).fetchall()
        return [LearningPolicy.model_validate(json.loads(r[0])) for r in rows]

    def activate_learning_policy(self, policy_id: str) -> None:
        self._conn.execute("UPDATE learning_policies SET is_active = FALSE WHERE is_active")
        self._conn.execute(
            "UPDATE learning_policies SET is_active = TRUE WHERE policy_id = ?", [policy_id],
        )
        logger.info("Learning policy activated", policy_id=policy_id)

    # ── Workspace Management ──

    def get_workspace_health(self, workspace_id: str) -> dict:
        row = self._conn.execute(
            "SELECT payload_json FROM memory_workspaces WHERE workspace_id = ?",
            [workspace_id],
        ).fetchone()
        if row is None:
            return {"error": "workspace not found"}
        ws = MemoryWorkspace.model_validate(json.loads(row[0]))
        return {
            "workspace_id": ws.workspace_id,
            "name": ws.name,
            "trade_count": ws.trade_count,
            "size_bytes": ws.size_bytes,
            "last_used": ws.last_used_at.isoformat() if ws.last_used_at else None,
            "is_active": ws.is_active,
        }

    def get_workspace_statistics(self, workspace_id: str) -> dict:
        ws = self.get_workspace_health(workspace_id)
        if "error" in ws:
            return ws
        return {
            **ws,
            "db_path": "",
        }

    def verify_workspace(self, workspace_id: str) -> dict:
        row = self._conn.execute(
            "SELECT payload_json, db_path FROM memory_workspaces WHERE workspace_id = ?",
            [workspace_id],
        ).fetchone()
        if row is None:
            return {"verified": False, "error": "workspace not found"}
        ws = MemoryWorkspace.model_validate(json.loads(row[0]))
        db_path = row[1]
        entries: list[str] = []
        try:
            import os
            if os.path.exists(db_path):
                entries.append("db_file_exists")
            else:
                entries.append("db_file_missing")
            if os.path.exists(db_path + ".wal"):
                entries.append("wal_exists")
        except Exception:
            entries.append("check_failed")
        return {
            "verified": "db_file_exists" in entries and "db_file_missing" not in entries,
            "workspace_id": workspace_id,
            "name": ws.name,
            "checks": entries,
        }

    def compare_workspaces(self, ws_id_a: str, ws_id_b: str) -> dict:
        a = self._conn.execute(
            "SELECT payload_json FROM memory_workspaces WHERE workspace_id = ?", [ws_id_a],
        ).fetchone()
        b = self._conn.execute(
            "SELECT payload_json FROM memory_workspaces WHERE workspace_id = ?", [ws_id_b],
        ).fetchone()
        if not a or not b:
            return {"error": "one or both workspaces not found"}
        wa = MemoryWorkspace.model_validate(json.loads(a[0]))
        wb = MemoryWorkspace.model_validate(json.loads(b[0]))
        return {
            "workspace_a": wa.name,
            "workspace_b": wb.name,
            "trade_count_a": wa.trade_count,
            "trade_count_b": wb.trade_count,
            "size_a_bytes": wa.size_bytes,
            "size_b_bytes": wb.size_bytes,
            "last_used_a": wa.last_used_at.isoformat() if wa.last_used_at else None,
            "last_used_b": wb.last_used_at.isoformat() if wb.last_used_at else None,
        }

    def export_workspace(self, workspace_id: str, export_path: str) -> str:
        row = self._conn.execute(
            "SELECT payload_json, db_path FROM memory_workspaces WHERE workspace_id = ?",
            [workspace_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"Workspace {workspace_id} not found")
        ws = MemoryWorkspace.model_validate(json.loads(row[0]))
        src_db = row[1]
        import shutil
        shutil.copy2(src_db, export_path)
        logger.info("Workspace exported", workspace_id=workspace_id, export_path=export_path)
        return export_path

    def import_workspace(self, import_path: str, name: str = "") -> str:
        import uuid
        ws_id = str(uuid.uuid4())
        db_path = f"data/{name or 'imported'}_{ws_id[:8]}.duckdb"
        import shutil
        shutil.copy2(import_path, db_path)
        ws = MemoryWorkspace(
            workspace_id=ws_id,
            name=name or f"Imported {ws_id[:8]}",
            db_path=db_path,
            is_active=False,
        )
        self.save_workspace(ws)
        logger.info("Workspace imported", workspace_id=ws_id, name=ws.name)
        return ws_id

    def archive_workspace(self, workspace_id: str) -> None:
        row = self._conn.execute(
            "SELECT db_path, name FROM memory_workspaces WHERE workspace_id = ?",
            [workspace_id],
        ).fetchone()
        if row is None:
            return
        db_path, name = row
        import shutil
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archived_path = f"{db_path}.archived.{ts}"
        import os
        if os.path.exists(db_path):
            shutil.move(db_path, archived_path)
        wal_path = db_path + ".wal"
        if os.path.exists(wal_path):
            shutil.move(wal_path, archived_path + ".wal")
        self._conn.execute("DELETE FROM memory_workspaces WHERE workspace_id = ?", [workspace_id])
        logger.info("Workspace archived", workspace_id=workspace_id, name=name, archived_path=archived_path)

    def clear_workspace(self, workspace_id: str) -> int:
        row = self._conn.execute(
            "SELECT db_path FROM memory_workspaces WHERE workspace_id = ?",
            [workspace_id],
        ).fetchone()
        if row is None:
            return 0
        db_path = row[0]
        import os
        if os.path.exists(db_path):
            try:
                import duckdb as _ddb
                tmp = _ddb.connect(db_path)
                tables = tmp.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                count = 0
                for t in tables:
                    tmp.execute(f"DELETE FROM {t[0]}")
                    count += 1
                tmp.close()
                logger.info("Workspace cleared", workspace_id=workspace_id, tables_emptied=count)
                return count
            except Exception as e:
                logger.warning("Failed to clear workspace", error=str(e))
                return 0
        return 0

    # ── Adaptive Feedback State ──

    def save_feedback_state(self, setting_name: str, value: str, reason: str = "") -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO adaptive_feedback_state
                (setting_name, value, reason, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            [setting_name, value, reason, datetime.utcnow().isoformat()],
        )

    def load_feedback_state(self) -> dict:
        rows = self._conn.execute(
            "SELECT setting_name, value FROM adaptive_feedback_state ORDER BY setting_name"
        ).fetchall()
        result: dict = {}
        for name, raw in rows:
            try:
                import json
                val = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                val = raw
            result[name] = val
        return result

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
