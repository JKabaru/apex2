from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.learning.observation import ObservationCategory, SourceComponent


def _generate_id() -> str:
    return str(uuid.uuid4())


class ObservationAggregate(BaseModel, frozen=True):
    """Compressed summary of multiple observations.

    Compression produces a distinct entity rather than mutating the
    original observations. The original immutable observations remain
    untouched and independently analyzable."""
    aggregate_id: str = Field(default_factory=_generate_id)
    observation_ids: list[str] = Field(default_factory=list)
    count: int
    window_start: datetime
    window_end: datetime
    source: SourceComponent
    category: ObservationCategory
    symbol: str
    importance: float
    summary_data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
