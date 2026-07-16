from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class HypothesisStatus(str, Enum):
    DRAFT = "draft"
    TESTING = "testing"
    STRENGTHENED = "strengthened"
    WEAKENED = "weakened"
    MATURE = "mature"
    DISCARDED = "discarded"


class Hypothesis(BaseModel, frozen=True):
    """Interpretation of one or more patterns within a timeline.

    A hypothesis is not an observation summary — it is a causal or
    explanatory statement about why a pattern occurred. Multiple
    hypotheses may coexist within the same timeline."""
    hypothesis_id: str = Field(default_factory=_generate_id)
    statement: str
    pattern_ids: list[str] = Field(default_factory=list)
    symbol: str
    timeframe: str
    side: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: HypothesisStatus = HypothesisStatus.DRAFT
    evidence_count: int = 0
    confidence: float = 0.0
    supporting_count: int = 0
    contradicting_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HypothesisEvidence(BaseModel, frozen=True):
    """Evidence linking an observation to a hypothesis.

    This is a directed edge in an evidence graph. The same observation
    may support one hypothesis and contradict another."""
    hypothesis_id: str
    timeline_id: str
    observation_id: str
    weight: float = 1.0
    supports: bool
    added_at: datetime = Field(default_factory=datetime.utcnow)
