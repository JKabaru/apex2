from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog

from src.recommendations.models import AdaptiveDecision, Finding
from src.recommendations.store import ConfigurationStore
from src.storage.learning.learning_corpus import LearningCorpus

logger = structlog.get_logger("profile_adapter")


class ProfileAdapter:
    """Bridges Beliefs → profile adaptation via the adaptive parameter system.

    Reads active beliefs from the learning corpus and maps them to
    AdaptiveDecision objects that tune risk/execution parameters.
    """

    BELIEF_TO_PARAM: dict[str, dict[str, Any]] = {
        "low_confidence_tendency": {
            "parameter_id": "min_llm_confidence",
            "config_path": "risk.min_llm_confidence",
            "min": 0.1,
            "max": 0.8,
            "default": 0.3,
            "adjustment_fn": lambda strength: round(0.3 + strength * 0.2, 2),
        },
        "critique_overturn_rate": {
            "parameter_id": "stop_loss_pct",
            "config_path": "execution.stop_loss_pct",
            "min": 0.90,
            "max": 0.99,
            "default": 0.98,
            "adjustment_fn": lambda strength: round(0.98 - strength * 0.03, 3),
        },
    }

    def __init__(self, corpus: LearningCorpus, config_store: ConfigurationStore):
        self._corpus = corpus
        self._config_store = config_store

    def adapt(self) -> list[dict[str, Any]]:
        """Read active beliefs, create/process adaptive decisions.

        Returns a list of action summaries.
        """
        beliefs = self._corpus.get_active_beliefs()
        if not beliefs:
            return []

        results: list[dict[str, Any]] = []
        active_profile = self._config_store.get_active_profile()
        if active_profile is None:
            logger.warning("PROFILE_ADAPT_SKIP_NO_ACTIVE_PROFILE")
            return []
        profile_id = active_profile.profile_id
        active_versions = self._config_store.get_active_adaptive_versions(profile_id)

        for belief in beliefs:
            # Symbol bias → finding
            if belief.category == "symbol_bias" and belief.confidence >= 0.4:
                finding = Finding(
                    category="symbol_bias",
                    description=belief.statement,
                    supporting_metrics={
                        "symbol": belief.symbol,
                        "strength": belief.strength,
                        "observation_count": belief.observation_count,
                        "dominant_action": belief.metadata.get("dominant_action", "UNKNOWN"),
                        "ratio": belief.metadata.get("ratio", 0.0),
                    },
                    severity="HIGH" if belief.strength > 0.6 else "MEDIUM",
                )
                self._config_store.save_finding(finding)
                results.append({
                    "action": "finding_created",
                    "category": "symbol_bias",
                    "symbol": belief.symbol,
                    "finding_id": finding.finding_id,
                })
                logger.info(
                    "PROFILE_ADAPT_SYMBOL_BIAS_FINDING",
                    symbol=belief.symbol,
                    finding_id=finding.finding_id,
                    dominant_action=belief.metadata.get("dominant_action", "UNKNOWN"),
                    ratio=belief.metadata.get("ratio", 0.0),
                    _force_log=True,
                )
                continue

            mapping = self.BELIEF_TO_PARAM.get(belief.category)
            if mapping is None:
                continue

            if belief.confidence < 0.4:
                logger.info(
                    "PROFILE_ADAPT_SKIP_LOW_CONFIDENCE",
                    belief_id=belief.belief_id,
                    category=belief.category,
                    confidence=belief.confidence,
                    _force_log=True,
                )
                continue

            param_id = mapping["parameter_id"]
            current_version = active_versions.get(param_id)
            current_value = current_version.value if current_version else mapping["default"]

            proposed_value = mapping["adjustment_fn"](belief.strength)
            proposed_value = max(mapping["min"], min(mapping["max"], proposed_value))

            if abs(proposed_value - current_value) < 0.005:
                logger.info(
                    "PROFILE_ADAPT_SKIP_NO_CHANGE",
                    belief_id=belief.belief_id,
                    parameter_id=param_id,
                    current=current_value,
                    _force_log=True,
                )
                continue

            existing = self._config_store.list_adaptive_decisions(status_filter="pending", limit=100)
            already_pending = any(d.parameter_id == param_id for d in existing)
            if already_pending:
                logger.info(
                    "PROFILE_ADAPT_SKIP_ALREADY_PENDING",
                    belief_id=belief.belief_id,
                    parameter_id=param_id,
                    _force_log=True,
                )
                continue

            decision = AdaptiveDecision(
                recommendation_id=f"belief_{uuid.uuid4().hex[:12]}",
                parameter_id=param_id,
                proposed_value=proposed_value,
                current_value=current_value,
                confidence=belief.confidence,
                sample_count=belief.observation_count,
                required_evidence_count=5,
                status="pending",
                reason=f"Belief-driven: {belief.category} (strength={belief.strength:.2f}, obs={belief.observation_count})",
                evidence_summary=belief.statement,
            )
            self._config_store.save_adaptive_decision(decision)

            logger.info(
                "PROFILE_ADAPT_DECISION_CREATED",
                belief_id=belief.belief_id,
                parameter_id=param_id,
                current_value=current_value,
                proposed_value=proposed_value,
                confidence=belief.confidence,
                observation_count=belief.observation_count,
                _force_log=True,
            )
            results.append({
                "action": "decision_created",
                "parameter_id": param_id,
                "from": current_value,
                "to": proposed_value,
                "belief_id": belief.belief_id,
                "decision_id": decision.decision_id,
            })

        if results:
            from src.recommendations.lifecycle import process_adaptive_decisions
            processed = process_adaptive_decisions(
                self._config_store, profile_id, auto_approve=True,
            )
            for p in processed:
                results.append({
                    "action": "decision_" + p["status"],
                    "parameter_id": p.get("parameter_id", ""),
                    "decision_id": p.get("decision_id", ""),
                })
            logger.info(
                "PROFILE_ADAPT_CYCLE",
                decisions_created=len([r for r in results if r["action"] == "decision_created"]),
                decisions_processed=len(processed),
                _force_log=True,
            )

        return results
