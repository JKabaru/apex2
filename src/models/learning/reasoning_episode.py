from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class ReasoningEpisode(BaseModel, frozen=True):
    """Persistent record of an LLM reasoning decision.

    Every LLM decision produces a ReasoningEpisode that captures what
    the LLM saw, what it chose, why it chose it, and what signals it
    weighed. This is the foundation for reflection, self-critique, and
    belief updates.
    """
    episode_id: str = Field(default_factory=_generate_id)
    decision_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    timeframe: str = ""
    prompt_hash: str = ""
    prompt_preview: str = ""

    market_summary: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    portfolio_summary: dict[str, Any] = Field(default_factory=dict)

    action: str = ""
    confidence: float = 0.0
    rationale: str = ""
    risk_assessment: str = ""
    llm_response_raw: str = ""

    chosen_signals: list[str] = Field(default_factory=list)
    ignored_signals: list[str] = Field(default_factory=list)
    predicted_outcome: str = ""

    retrieval_memory_ids: list[str] = Field(default_factory=list)
    retrieval_belief_ids: list[str] = Field(default_factory=list)
    retrieval_knowledge_ids: list[str] = Field(default_factory=list)

    execution_id: str = ""
    correlation_id: str = ""
    opportunity_id: str = ""
    strategy_version: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)
