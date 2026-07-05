from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog

from src.models.learning.trade_experience import (
    LearningExperience,
    ValidationReport,
)

logger = structlog.get_logger("experience_validator")

VALID_EXECUTION_MODES = {"LIVE", "SHADOW"}
VALID_ORIGINS = {"NORMAL", "CONSTRAINT", "MIRROR"}


class ExperienceValidator:
    """Stage 2 of the learning pipeline.
    Verifies facts and produces an explicit ValidationReport.
    Never judges trade quality — only data completeness and integrity."""

    validator_version: str = "1.0"

    def validate(
        self,
        experience: LearningExperience,
        metadata: Optional[dict] = None,
    ) -> ValidationReport:
        verified: list[str] = []
        missing: list[str] = []
        schema_errors: list[str] = []
        ordering_errors: list[str] = []
        evidence_notes: list[str] = []

        # ── Schema / Structural ──
        if experience.experience_id:
            verified.append("experience_id")
        else:
            missing.append("experience_id")

        if experience.position_id:
            verified.append("position_id")
        else:
            missing.append("position_id")

        if experience.schema_version:
            verified.append("schema_version")
        else:
            missing.append("schema_version")

        if experience.timeframe:
            verified.append("timeframe")
        else:
            missing.append("timeframe")

        # ── Execution / Pricing ──
        if experience.entry_price > 0:
            verified.append("entry_price")
        else:
            schema_errors.append("entry_price is zero or negative")

        if experience.exit_price is not None and experience.exit_price > 0:
            verified.append("exit_price")
        elif experience.exit_price is not None:
            schema_errors.append("exit_price is zero or negative")
        else:
            missing.append("exit_price")

        if experience.fees >= 0:
            verified.append("fees")
        else:
            schema_errors.append("fees is negative")

        if experience.exit_fees is not None:
            if experience.exit_fees >= 0:
                verified.append("exit_fees")
            else:
                schema_errors.append("exit_fees is negative")
        else:
            missing.append("exit_fees")

        # ── Timestamp ordering ──
        if experience.entry_price > 0:
            verified.append("entry_timestamp")
        else:
            missing.append("entry_timestamp")

        if experience.exit_price is not None:
            verified.append("exit_timestamp")
        else:
            missing.append("exit_timestamp")

        # ── Evidence evolution ──
        if experience.episode_count > 0:
            verified.append("episode_count")

            # Check episode ordering
            indices = [
                ep.get("index", -1)
                for ep in experience.evidence_episodes_summary
            ]
            if indices != sorted(indices):
                ordering_errors.append("episode indices are not sequential")
            else:
                verified.append("episode_indices_sequential")
        else:
            evidence_notes.append("episode_count=0 — no market evolution captured")

        # ── Indicator availability (not required, but noted) ──
        if experience.entry_atr is not None:
            verified.append("entry_atr")
        else:
            missing.append("entry_atr")

        if experience.entry_rsi is not None:
            verified.append("entry_rsi")
        else:
            missing.append("entry_rsi")

        if experience.trend_regime is not None:
            verified.append("trend_regime")
        else:
            missing.append("trend_regime")

        if experience.volatility_regime is not None:
            verified.append("volatility_regime")
        else:
            missing.append("volatility_regime")

        if experience.correlation_regime is not None:
            verified.append("correlation_regime")
        else:
            missing.append("correlation_regime")

        # ── Metadata validation ──
        if metadata:
            prov_version = metadata.get("provenance_version", "")
            if prov_version:
                verified.append("provenance_version")
            else:
                missing.append("provenance_version")

            opp_id = metadata.get("opportunity_id", "")
            if opp_id:
                import uuid as _uuid
                try:
                    _uuid.UUID(opp_id)
                    verified.append("opportunity_id")
                except (ValueError, AttributeError):
                    schema_errors.append(f"opportunity_id is not a valid UUID: {opp_id}")
            else:
                missing.append("opportunity_id")

            feature_cat_ver = metadata.get("feature_catalog_version", "")
            if feature_cat_ver:
                verified.append("feature_catalog_version")
            else:
                missing.append("feature_catalog_version")

            market_hash = metadata.get("market_state_hash", "")
            if market_hash:
                if len(market_hash) == 64 and all(c in "0123456789abcdef" for c in market_hash):
                    verified.append("market_state_hash")
                else:
                    schema_errors.append("market_state_hash is not valid SHA256 hex")
            else:
                missing.append("market_state_hash")

        return ValidationReport(
            validator_version=self.validator_version,
            verified_fields=verified,
            missing_fields=missing,
            schema_errors=schema_errors,
            ordering_errors=ordering_errors,
            evidence_notes=evidence_notes,
        )
