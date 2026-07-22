from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from src.models.tim.enums import (
    JournalEventType,
    OriginQuality,
    ProtectionMode,
    ThesisStatus,
    TradeMemoryRecoveryState,
)


def _generate_id() -> str:
    return str(uuid.uuid4())


_THESIS_TRANSITIONS: dict[ThesisStatus, set[ThesisStatus]] = {
    ThesisStatus.INTACT: {
        ThesisStatus.STRENGTHENED,
        ThesisStatus.EVOLVED,
        ThesisStatus.WEAKENED,
        ThesisStatus.AT_RISK,
        ThesisStatus.INVALIDATED,
    },
    ThesisStatus.STRENGTHENED: {
        ThesisStatus.EVOLVED,
        ThesisStatus.WEAKENED,
        ThesisStatus.AT_RISK,
        ThesisStatus.INVALIDATED,
    },
    ThesisStatus.EVOLVED: {
        ThesisStatus.WEAKENED,
        ThesisStatus.AT_RISK,
        ThesisStatus.INVALIDATED,
    },
    ThesisStatus.WEAKENED: {
        ThesisStatus.AT_RISK,
        ThesisStatus.INVALIDATED,
        ThesisStatus.INTACT,
    },
    ThesisStatus.AT_RISK: {
        ThesisStatus.WEAKENED,
        ThesisStatus.INVALIDATED,
    },
    ThesisStatus.INVALIDATED: set(),
}


class TradeOrigin(BaseModel):
    memory_id: str = Field(default_factory=_generate_id)
    position_id: str
    origin_episode_id: str
    origin_quality: OriginQuality = OriginQuality.UNKNOWN
    entry_thesis: str = ""
    entry_price: float = 0.0
    entry_atr: Optional[float] = None
    entry_timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    side: str
    anchor_symbol: str = ""
    timeframe: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkingMemory(BaseModel):
    memory_id: str
    position_id: str
    version: int = Field(default=1, ge=1)
    checksum: str = ""
    thesis_status: ThesisStatus = ThesisStatus.INTACT
    protection_mode: ProtectionMode = ProtectionMode.MECHANICAL_ONLY
    current_stop: Optional[float] = None
    current_target: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    mae_atr_multiple: Optional[float] = None
    mfe_atr_multiple: Optional[float] = None
    review_count: int = 0
    failed_review_count: int = 0
    last_review_timestamp: Optional[datetime] = None
    next_review_conditions: str = "{}"
    watchdog_timer_start: Optional[datetime] = None
    watchdog_deadline: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_checksum(self) -> WorkingMemory:
        if self.version > 1 and not self.checksum:
            raise ValueError("checksum required when version > 1")
        return self


class TradeJournalEntry(BaseModel):
    journal_id: str = Field(default_factory=_generate_id)
    position_id: str
    memory_id: str
    version: int = Field(ge=1)
    event_type: JournalEventType
    event_data: str = "{}"
    previous_checksum: Optional[str] = None
    new_checksum: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class JournalCompressionSummary(BaseModel):
    compression_id: str = Field(default_factory=_generate_id)
    position_id: str
    memory_id: str
    version_start: int
    version_end: int
    summary_data: str = "{}"
    original_entry_count: int
    compressed_at: datetime = Field(default_factory=datetime.utcnow)
    compression_version: str = "1.0"


class TradeMemoryRecoveryRecord(BaseModel):
    memory_id: str = Field(default_factory=_generate_id)
    position_id: str
    recovery_state: TradeMemoryRecoveryState = TradeMemoryRecoveryState.NOT_REQUIRED
    recovered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
