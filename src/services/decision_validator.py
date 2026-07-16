from __future__ import annotations

import json
import re
import structlog

from pydantic import ValidationError

from src.intelligence.models import PromptContext
from src.models.reasoning import LLMDecision
from src.services.evidence_policy import get_policy

logger = structlog.get_logger("decision_validator")


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def validate(raw_json: str, evidence: PromptContext) -> LLMDecision:
    try:
        stripped = raw_json.strip()
        if not stripped:
            logger.warning("Empty LLM response")
            return _fallback("Empty LLM response")

        json_str = _extract_json(stripped)
        parsed = json.loads(json_str)
        decision = LLMDecision.model_validate(parsed)

        logger.info(
            "LLM decision validated",
            action=decision.action,
            confidence=decision.confidence,
            source=evidence.evidence_source,
            rationale=decision.rationale[:500] if decision.rationale else "",
            risk_assessment=decision.risk_assessment[:300] if decision.risk_assessment else "",
        )
        return decision

    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("LLM response validation failed", error=str(e), raw_preview=raw_json[:200])
        return _fallback(f"Validation failed: {e}")

    except Exception as e:
        logger.error("Unexpected validation error", error=str(e))
        return _fallback(f"Unexpected error: {e}")


def apply_ceiling(decision: LLMDecision, evidence: PromptContext) -> LLMDecision:
    policy = get_policy(evidence.evidence_source)
    if decision.confidence > policy.max_confidence:
        logger.info(
            "Confidence ceiling applied",
            tier=evidence.evidence_tier,
            source=evidence.evidence_source,
            policy_label=policy.label,
            original_conf=decision.confidence,
            capped_conf=policy.max_confidence,
        )
        return LLMDecision(
            action=decision.action,
            confidence=policy.max_confidence,
            rationale=decision.rationale,
            risk_assessment=decision.risk_assessment,
        )
    return decision


def _fallback(reason: str) -> LLMDecision:
    logger.warning("Returning fallback HOLD decision", reason=reason)
    return LLMDecision(
        action="HOLD",
        confidence=0.0,
        rationale=f"Decision fallback: {reason[:250]}",
        risk_assessment="",
    )
