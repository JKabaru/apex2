from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


class SourceComponent(str, Enum):
    SCANNER = "scanner"
    TRADE_COORDINATOR = "trade_coordinator"
    POSITION_MANAGER = "position_manager"
    RISK_MANAGER = "risk_manager"
    EXECUTION = "execution"
    LIFECYCLE = "lifecycle"
    SESSION_INTELLIGENCE = "session_intelligence"


class ObservationCategory(str, Enum):
    PRICE_ACTION = "price_action"
    VOLATILITY = "volatility"
    REGIME = "regime"
    POSITION = "position"
    EXECUTION = "execution"
    RISK = "risk"
    SIGNAL = "signal"
    MARKET = "market"
    SESSION = "session"
    SYSTEM = "system"


class Observation(BaseModel, frozen=True):
    """Immutable historical fact. Never modified after creation.

    An observation records something that happened at a point in time.
    It has no built-in relationship to any position or timeline --- those
    associations are established through TimelineObservation.

    Position-specific context (e.g. which position was affected) belongs
    in the ``data`` dict, not as a structural field."""
    observation_id: str = Field(default_factory=_generate_id)
    timestamp: datetime
    source: SourceComponent
    category: ObservationCategory
    importance: float
    symbol: str
    data: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
