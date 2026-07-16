from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class PatternCategory(str, Enum):
    FAILED_BREAKOUT = "failed_breakout"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    TRAILING_STOP_OSCILLATION = "trailing_stop_oscillation"
    LATE_EXECUTION = "late_execution"
    REPEATED_PROTECTION_RETRY = "repeated_protection_retry"
    PRICE_REJECTION = "price_rejection"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    POSITION_OSCILLATION = "position_oscillation"


class Pattern(BaseModel, frozen=True):
    """An objective pattern detected within a timeline.

    Patterns sit between Timelines and Hypotheses in the memory hierarchy.
    A timeline can contain many patterns. A hypothesis interprets one or
    more patterns.

    Patterns are objective (they describe what happened).
    Hypotheses are subjective (they interpret why it happened)."""
    pattern_id: str = Field(default_factory=_generate_id)
    timeline_id: str
    category: PatternCategory
    description: str
    observation_ids: list[str] = Field(default_factory=list)
    start_time: datetime
    end_time: datetime
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
