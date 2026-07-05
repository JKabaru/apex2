from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import duckdb
import structlog

from src.recommendations.models import (
    ActivationRecord,
    ConfigurationProfile,
    Finding,
    Intervention,
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
            CREATE INDEX IF NOT EXISTS idx_profiles_active
            ON profiles (is_active)
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
            INSERT OR IGNORE INTO profiles
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
            "SELECT payload_json FROM profiles WHERE is_active = TRUE LIMIT 1",
        ).fetchone()
        if row is None:
            return None
        return ConfigurationProfile.model_validate(json.loads(row[0]))

    def activate_profile(
        self,
        profile_id: str,
        activated_by: str = "system",
    ) -> None:
        # Deactivate current active
        self._conn.execute(
            "UPDATE profiles SET is_active = FALSE WHERE is_active = TRUE",
        )
        # Mark new profile active
        self._conn.execute(
            "UPDATE profiles SET is_active = TRUE WHERE profile_id = ?",
            [profile_id],
        )
        # Deactivate old activation records for this profile
        self._conn.execute(
            "UPDATE activation_history SET deactivated_at = ? WHERE profile_id = ? AND deactivated_at IS NULL",
            [datetime.utcnow().isoformat(), profile_id],
        )
        # Create activation record
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

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
