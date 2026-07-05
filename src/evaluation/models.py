from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class DecisionCapture(BaseModel, frozen=True):
    """Immutable record of what the LLM decided and what evidence was available.
    Captured once when CANDIDATE_EVALUATED is published. Never modified."""
    opportunity_id: str
    candidate_id: str
    symbol: str
    llm_action: str
    llm_confidence: float
    llm_rationale: str
    llm_risk_assessment: str
    evidence_source: str
    evidence_tier: int
    captured_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionEvaluation(BaseModel, frozen=True):
    """Post-trade evaluation of an LLM decision against actual outcomes.
    Immutable, deterministic, append-only. Self-contained — no external lookups required."""
    evaluation_id: str = Field(default_factory=_generate_id)
    position_id: str
    opportunity_id: str
    candidate_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = "1.0"

    # ── Decision inputs (copied from capture for self-containment) ──
    llm_action: str
    llm_confidence: float
    llm_rationale: str
    llm_risk_assessment: str
    evidence_source: str
    evidence_tier: int

    # ── Actual outcomes (copied from manifest + position) ──
    actual_side: str
    actual_quantity: float
    actual_entry_price: float
    actual_exit_price: Optional[float]
    actual_pnl: Optional[float]
    actual_pnl_atr: Optional[float]
    actual_duration_minutes: Optional[float]
    actual_max_drawdown: float
    actual_highest_profit: float
    actual_exit_reason: Optional[str]
    actual_integrity_score: int

    # ── Evaluation metrics ──
    was_profitable: Optional[bool]
    action_aligned: bool
    confidence_vs_outcome: str
    evaluation_notes: list[str] = Field(default_factory=list)
