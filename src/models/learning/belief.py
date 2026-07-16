from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class Belief(BaseModel, frozen=True):
    """Meta-cognitive insight about the agent's own decision-making quality.

    Unlike Hypothesis (market patterns) and Knowledge (verified conclusions),
    Beliefs capture *behavioral* insights: confidence tendencies, symbol
    biases, critique patterns — used to adapt the agent's own profile.
    """
    belief_id: str = Field(default_factory=_generate_id)
    statement: str
    category: str  # e.g. "low_confidence_tendency", "symbol_bias", "critique_overturn_rate"
    symbol: str = ""
    confidence: float = 0.0  # how strongly we hold this belief (0-1)
    strength: float = 0.0    # signal strength from evidence (0-1)
    source: str = "reflection"  # what generated this belief
    observation_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    deprecated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
