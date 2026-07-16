from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class TimelineStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    ANALYZED = "analyzed"
    ARCHIVED = "archived"


class Timeline(BaseModel, frozen=True):
    """Event-driven record of a position's lifecycle.

    A timeline records history — it does not own prediction state.
    Prediction is modelled independently (see PredictionLifecycle in
    Phase B). A position has exactly one timeline.

    Lifecycle: OPEN -> CLOSED -> READY_FOR_ANALYSIS -> ANALYZED -> ARCHIVED"""
    timeline_id: str = Field(default_factory=_generate_id)
    position_id: str
    symbol: str
    side: str
    timeframe: str
    opened_at: datetime
    closed_at: datetime | None = None
    status: TimelineStatus = TimelineStatus.OPEN
    observation_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineObservation(BaseModel, frozen=True):
    """Link between a timeline and an observation.

    The canonical association. One observation may appear in multiple
    timelines (e.g. a volatility event affecting several positions),
    and one timeline aggregates many observations."""
    timeline_id: str
    observation_id: str
    sequence: int
    added_at: datetime = Field(default_factory=datetime.utcnow)
    importance_at_addition: float
