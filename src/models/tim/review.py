from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.tim.enums import ProtectionMode, ReviewStatus, ReviewTriggerType, ThesisStatus, TriggerType


def _generate_id() -> str:
    return str(uuid.uuid4())


class ReviewConditions(BaseModel):
    reference_price: Optional[float] = None
    reference_atr: Optional[float] = None
    reference_stop: Optional[float] = None
    reference_target: Optional[float] = None
    reference_trend: Optional[str] = None
    reference_volatility: Optional[str] = None
    min_price_delta_pct: Optional[float] = None
    max_time_delta_minutes: Optional[float] = None
    reference_timestamp: Optional[datetime] = None


class ReviewSession(BaseModel):
    session_id: str = Field(default_factory=_generate_id)
    position_id: str
    memory_id: str
    trigger_type: ReviewTriggerType
    trigger_id: Optional[str] = None
    status: ReviewStatus = ReviewStatus.PENDING
    thesis_status_before: Optional[ThesisStatus] = None
    thesis_status_after: Optional[ThesisStatus] = None
    protection_mode_before: ProtectionMode = ProtectionMode.MECHANICAL_ONLY
    protection_mode_after: Optional[ProtectionMode] = None
    intent_envelope_id: Optional[str] = None
    fallback_used: bool = False
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    conditions_snapshot: str = "{}"


class ReviewTrigger(BaseModel):
    trigger_id: str = Field(default_factory=_generate_id)
    position_id: str
    trigger_type: TriggerType
    trigger_reason: str = ""
    trigger_value: Optional[float] = None
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    schedule_id: Optional[str] = None
    suppressed: bool = False


class ReviewSchedule(BaseModel):
    schedule_id: str = Field(default_factory=_generate_id)
    position_id: str
    memory_id: str
    trigger_type: ReviewTriggerType
    status: str = "ACTIVE"
    conditions: str = "{}"
    interval_minutes: Optional[float] = None
    next_due_at: datetime
    last_fired_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TriggerSuppressionRecord(BaseModel):
    suppression_id: str = Field(default_factory=_generate_id)
    position_id: str
    trigger_id: str
    suppressed_at: datetime = Field(default_factory=datetime.utcnow)
    suppressed_by: str
    reason: str = ""
