from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.tim.enums import IntentStatus, IntentType


def _generate_id() -> str:
    return str(uuid.uuid4())


class TradeManagementIntent(BaseModel):
    intent_id: str = Field(default_factory=_generate_id)
    position_id: str
    memory_id: str
    intent_type: IntentType
    parameters: str = "{}"
    rationale: str = ""
    confidence: float = 0.0
    proposed_at: datetime = Field(default_factory=datetime.utcnow)
    status: IntentStatus = IntentStatus.PROPOSED
    processed_at: Optional[datetime] = None


class StrategicIntentEnvelope(BaseModel):
    envelope_id: str = Field(default_factory=_generate_id)
    position_id: str
    review_session_id: str
    intents: list[TradeManagementIntent] = Field(default_factory=list)
    llm_raw_response: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IntentExecutionRecord(BaseModel):
    execution_id: str = Field(default_factory=_generate_id)
    intent_id: str
    position_id: str
    intent_type: IntentType
    status: IntentStatus = IntentStatus.QUEUED
    attempt_number: int = 1
    max_attempts: int = 3
    client_order_id: Optional[str] = None
    stop_client_order_id: Optional[str] = None
    tp_client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
