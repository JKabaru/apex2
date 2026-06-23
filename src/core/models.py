from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PositionState(str, enum.Enum):
    DISCOVERED = "DISCOVERED"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    UNMANAGED_ADOPTED = "UNMANAGED_ADOPTED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ARCHIVED = "ARCHIVED"


class Position(BaseModel):
    position_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: str
    quantity: float
    avg_fill_price: float
    fees: float = 0.0
    exchange_order_ids: list[str] = Field(default_factory=list)
    entry_timestamp: datetime = Field(default_factory=datetime.utcnow)
    exit_timestamp: Optional[datetime] = None
    entry_thesis: str = ""
    anchor_symbol: str
    correlation_score: float = 0.0
    initial_stop_loss: float = 0.0
    initial_take_profit: float = 0.0
    current_stop: float = 0.0
    current_target: float = 0.0
    highest_unrealized_profit: float = 0.0
    maximum_drawdown: float = 0.0
    review_count: int = 0
    current_recommendation: Optional[str] = None
    lifecycle_state: PositionState = PositionState.DISCOVERED
    exit_reason: Optional[str] = None


class CandidateTrade(BaseModel):
    symbol: str
    anchor_symbol: str
    correlation_score: float = 0.0
    signal_strength: float = 0.0
    proposed_side: str
    proposed_quantity: float = 0.0


class SystemEvent(BaseModel):
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service_name: str
    payload: dict = Field(default_factory=dict)
