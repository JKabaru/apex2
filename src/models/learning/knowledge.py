from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class KnowledgeConfidence(str, Enum):
    EMERGING = "emerging"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    DEPRECATED = "deprecated"


class Knowledge(BaseModel, frozen=True):
    """Verified conclusion promoted from cross-hypothesis evidence.

    Knowledge emerges from hypotheses, not directly from timelines.
    It is the highest learned abstraction below adaptive parameters.

    Confidence levels:
      EMERGING   — 1-2 supporting hypotheses
      DEVELOPING — 3-4 supporting hypotheses
      ESTABLISHED — 5+ supporting, low contradiction
      DEPRECATED — previously established, now contradicted"""
    knowledge_id: str = Field(default_factory=_generate_id)
    statement: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    symbol: str
    timeframe: str
    confidence: KnowledgeConfidence = KnowledgeConfidence.EMERGING
    confidence_score: float = 0.0
    supporting_hypothesis_count: int = 0
    contradicting_hypothesis_count: int = 0
    cross_timeline_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    deprecated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
