from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

import structlog
from pydantic import BaseModel, Field

from src.models.reasoning import LLMDecision, PortfolioSnapshot

logger = structlog.get_logger("self_critique")


class CritiqueVerdict(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    OVERTURN = "OVERTURN"


class SelfCritiqueResult(BaseModel, frozen=True):
    verdict: CritiqueVerdict
    revised_action: Optional[Literal["BUY", "SELL", "HOLD", "ABSTAIN"]] = None
    revised_confidence: Optional[float] = None
    revised_rationale: Optional[str] = None
    revised_risk_assessment: Optional[str] = None
    critique_rationale: str = ""


class SelfCritiqueEngine:
    """Lightweight rule-based self-critique of LLM decisions.

    Checks confidence thresholds, evidence quality, and risk assessment
    completeness.  Verdicts:
      - PASS      → accept the decision as-is
      - REVISE    → adjust confidence / rationale / risk assessment
      - OVERTURN  → flip the action (e.g. BUY → HOLD)
    """

    def critique(
        self,
        decision: LLMDecision,
        portfolio: PortfolioSnapshot,
        *,
        evidence_source: str = "NONE",
        evidence_tokens: int = 0,
    ) -> SelfCritiqueResult:
        if decision.action in ("HOLD", "ABSTAIN"):
            return SelfCritiqueResult(
                verdict=CritiqueVerdict.PASS,
                critique_rationale="No-action decisions bypass critique",
            )

        checks: list[str] = []
        overturn = False
        revise_action: Optional[Literal["BUY", "SELL", "HOLD", "ABSTAIN"]] = None
        revise_conf: Optional[float] = None
        revise_rationale: Optional[str] = None
        revise_risk: Optional[str] = None

        min_conf = portfolio.min_llm_confidence
        if decision.confidence < min_conf:
            checks.append(
                f"confidence {decision.confidence:.2f} < threshold {min_conf:.2f}"
            )
            overturn = True
            revise_action = "HOLD"
            revise_conf = 0.0
            revise_rationale = (
                f"CRITIQUE_OVERTURN: confidence {decision.confidence:.2f} "
                f"below minimum {min_conf:.2f}. Original: {decision.rationale}"
            )
            revise_risk = "CRITIQUE_OVERTURN: confidence below threshold"

        if not overturn:
            risk_len = len(decision.risk_assessment.strip())
            if risk_len < 10:
                checks.append(f"risk_assessment too short ({risk_len} chars)")
                revise_risk = (
                    f"{decision.risk_assessment} "
                    f"[CRITIQUE_REVISE: risk assessment was incomplete]"
                )

            rationale_len = len(decision.rationale.strip())
            if rationale_len < 20:
                checks.append(f"rationale too short ({rationale_len} chars)")

            if revise_risk or revise_rationale:
                revise_conf = decision.confidence * 0.9

        if not checks:
            return SelfCritiqueResult(
                verdict=CritiqueVerdict.PASS,
                critique_rationale="All checks passed",
            )

        if overturn:
            logger.info(
                "SELF_CRITIQUE_OVERTURN",
                original_action=decision.action,
                revised_action=revise_action,
                reasons=checks,
                _force_log=True,
            )
            return SelfCritiqueResult(
                verdict=CritiqueVerdict.OVERTURN,
                revised_action=revise_action,
                revised_confidence=revise_conf,
                revised_rationale=revise_rationale,
                revised_risk_assessment=revise_risk,
                critique_rationale="; ".join(checks),
            )

        logger.info(
            "SELF_CRITIQUE_REVISE",
            original_action=decision.action,
            reasons=checks,
            _force_log=True,
        )
        return SelfCritiqueResult(
            verdict=CritiqueVerdict.REVISE,
            revised_confidence=revise_conf,
            revised_rationale=revise_rationale,
            revised_risk_assessment=revise_risk,
            critique_rationale="; ".join(checks),
        )
