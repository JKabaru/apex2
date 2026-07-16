from __future__ import annotations

from typing import Optional

import structlog

import json
import uuid
from datetime import datetime

from src.recommendations.models import (
    AdaptiveDecision,
    AdaptiveParameter,
    AdaptiveVersion,
)
from src.recommendations.store import ConfigurationStore

logger = structlog.get_logger("adaptive_lifecycle")


def merge_adaptive_config(
    config: dict,
    active_versions: dict[str, AdaptiveVersion],
    parameter_defs: dict[str, AdaptiveParameter],
) -> dict:
    """Merge active adaptive versions into the runtime config dict.

    For each active version whose parameter has a `config_path`, navigate
    into the config dict and set the value. Returns a new dict with merged values.
    """
    merged = {k: v for k, v in config.items()}
    for param_id, version in active_versions.items():
        param_def = parameter_defs.get(param_id)
        if param_def is None or not param_def.config_path:
            continue
        parts = param_def.config_path.split(".")
        target = merged
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        old_val = target.get(parts[-1])
        target[parts[-1]] = version.value
        logger.info(
            "Config merged from adaptive version",
            parameter_id=param_id,
            config_path=param_def.config_path,
            old_value=old_val,
            new_value=version.value,
        )
    return merged


def process_adaptive_decisions(
    config_store: ConfigurationStore,
    profile_id: str,
    auto_approve: bool = True,
) -> list[dict]:
    """Evaluate pending adaptive decisions and approve/apply them if eligible.

    Returns a list of processed decision summaries.
    """
    decisions = config_store.list_adaptive_decisions(status_filter="pending")
    results: list[dict] = []
    active_versions = config_store.get_active_adaptive_versions(profile_id)

    for decision in decisions:
        current = active_versions.get(decision.parameter_id)
        sufficient_evidence = decision.sample_count >= decision.required_evidence_count

        if not sufficient_evidence:
            logger.info(
                "AdaptiveDecision skipped — insufficient evidence",
                decision_id=decision.decision_id,
                parameter_id=decision.parameter_id,
                sample_count=decision.sample_count,
                required=decision.required_evidence_count,
            )
            config_store.update_decision_status(
                decision.decision_id, "rejected",
                reason=f"Insufficient evidence: {decision.sample_count}/{decision.required_evidence_count}",
            )
            results.append({"decision_id": decision.decision_id, "status": "rejected", "reason": "insufficient_evidence"})
            continue

        if not auto_approve:
            results.append({"decision_id": decision.decision_id, "status": "pending", "reason": "manual_review_required"})
            continue

        # Approve
        config_store.update_decision_status(
            decision.decision_id, "approved",
            reason="Auto-approved: sufficient evidence and confidence",
        )

        previous_value = current.value if current else decision.current_value
        version = AdaptiveVersion(
            parameter_id=decision.parameter_id,
            profile_id=profile_id,
            value=decision.proposed_value,
            previous_value=previous_value,
            confidence=decision.confidence,
            sample_count=decision.sample_count,
            required_evidence_count=decision.required_evidence_count,
            reason=f"Applied from AdaptiveDecision {decision.decision_id}",
            source="adaptive_lifecycle",
        )
        config_store.supersede_adaptive_version(decision.parameter_id, profile_id)
        config_store.save_adaptive_version(version)

        config_store.update_decision_status(
            decision.decision_id, "applied",
            reason=f"Applied as version {version.version_id}",
        )

        logger.info(
            "AdaptiveDecision applied",
            decision_id=decision.decision_id,
            parameter_id=decision.parameter_id,
            proposed_value=decision.proposed_value,
            version_id=version.version_id,
        )
        results.append({
            "decision_id": decision.decision_id,
            "status": "applied",
            "parameter_id": decision.parameter_id,
            "old_value": previous_value,
            "new_value": decision.proposed_value,
            "version_id": version.version_id,
        })

    return results


def trigger_adaptive_recommendation(
    config_store: ConfigurationStore,
    manifest: Any = None,
    confidence: float = 0.0,
    evidence_count: int = 0,
) -> bool:
    """Trigger adaptive recommendation lifecycle after a successful memory persist.
    Creates an AdaptiveDecision if any parameter has sufficient evidence.

    Returns True if at least one decision was created.
    """
    params = config_store.get_all_adaptive_parameters()
    if not params:
        return False

    triggered = False
    active_profile = config_store.get_active_profile()
    profile_id = active_profile.profile_id if active_profile else "default"

    for param_id, param_def in params.items():
        if evidence_count < param_def.required_evidence_count:
            continue

        active_versions = config_store.get_active_adaptive_versions(profile_id)
        current_version = active_versions.get(param_id)
        current_value = current_version.value if current_version else param_def.default_value

        proposed_value = current_value
        if confidence > 0.6 and evidence_count >= param_def.required_evidence_count:
            if param_id in ("stop_loss_pct", "take_profit_pct"):
                proposed_value = round(current_value * (1.0 + 0.01 * (confidence - 0.5)), 4)
            elif param_id in ("slippage_bps", "fee_bps", "spread_bps"):
                proposed_value = round(current_value * (1.0 - 0.01 * (confidence - 0.5)), 4)

        if proposed_value == current_value:
            continue

        existing = config_store.list_adaptive_decisions(status_filter="pending", limit=100)
        already_pending = any(
            d.parameter_id == param_id for d in existing
        )
        if already_pending:
            continue

        decision = AdaptiveDecision(
            recommendation_id=f"auto_{uuid.uuid4().hex[:12]}",
            parameter_id=param_id,
            proposed_value=proposed_value,
            current_value=current_value,
            confidence=confidence,
            sample_count=evidence_count,
            required_evidence_count=param_def.required_evidence_count,
            status="pending",
            reason=f"Auto-triggered after memory persist (evidence={evidence_count}, confidence={confidence:.3f})",
            evidence_summary=f"confidence={confidence:.3f}, evidence_count={evidence_count}",
        )
        config_store.save_adaptive_decision(decision)
        triggered = True
        logger.info(
            "Adaptive recommendation created",
            parameter_id=param_id,
            current_value=current_value,
            proposed_value=proposed_value,
            evidence_count=evidence_count,
            confidence=confidence,
            decision_id=decision.decision_id,
        )

    return triggered
