from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

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


class RiskDecision(str, enum.Enum):
    APPROVED = "APPROVED"
    REJECTED_CONSTRAINT = "REJECTED_CONSTRAINT"
    REJECTED_QUALITY = "REJECTED_QUALITY"
    DEFERRED = "DEFERRED"


def _generate_id() -> str:
    return str(uuid.uuid4())


class ExecutionContext(BaseModel, frozen=True):
    correlation_id: str
    execution_id: str
    trade_group_id: str
    candidate_id: str
    strategy_version: str = "1.0"
    llm_request_id: Optional[str] = None
    active_profile_id: Optional[str] = None
    session_id: Optional[str] = None

    execution_mode: str
    origin: str

    symbol: str
    timeframe: str = "5m"
    opportunity_id: str = ""
    side: str
    quantity: float
    anchor_symbol: str
    correlation_score: float = 0.0
    entry_thesis: str = ""
    llm_confidence: float = 0.0

    risk_decision: str = ""
    risk_decision_reason: str = ""

    execution_model: str = "fixed_friction_v1"
    execution_model_version: str = "1.0"
    execution_parameters: dict = Field(default_factory=lambda: {
        "spread_bps": 2.0,
        "fee_bps": 4.0,
        "slippage_bps": 3.0,
    })

    entry_timestamp: datetime = Field(default_factory=datetime.utcnow)

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", False)
        return super().model_dump(*args, **kwargs)


class Difference(BaseModel):
    """Structured diff. Never store human-readable strings as canonical data."""
    field: str
    previous: Any
    current: Any


class TradeContext(BaseModel):
    """Immutable after entry. Represents the exact market context that
    justified opening this position. Never modified once assigned."""
    anchor_symbol: str
    target_symbol: str = ""
    relationship: str = ""
    direction: str
    thesis: str = ""
    expected_catalyst: str = ""
    expected_invalidation: str = ""
    expected_opportunity: str = ""
    expected_holding_horizon_hours: float = 0.0
    timeframe: str = "5m"
    scanner_name: str = ""
    scanner_version: str = ""
    strategy_name: str = ""
    strategy_version: str = ""
    opportunity_timestamp: Optional[datetime] = None


class InitialEvidence(BaseModel):
    """Immutable. Captured once at entry. Never overwritten."""
    price: float
    rsi: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = None
    trend_regime: str = "UNKNOWN"
    volume_profile: str = "UNKNOWN"
    volatility_regime: str = "UNKNOWN"
    momentum: str = "UNKNOWN"
    correlation_regime: str = "UNKNOWN"
    correlation_score: float = 0.0
    entry_timestamp: datetime
    integrity: str = "HIGH"
    source: str = "original"


class MarketEvidence(BaseModel):
    """Complete market state at a point in time.
    Generated only when a categorical state field changes."""
    episode_id: str
    timestamp: datetime
    price: float
    rsi: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = None
    trend_regime: str
    volume_profile: str
    volatility_regime: str
    momentum: str
    correlation_regime: str
    correlation_score: float = 0.0
    drift_from_entry: list[Difference] = Field(default_factory=list)
    change_since_last_cycle: list[Difference] = Field(default_factory=list)
    integrity: str = "HIGH"


class EvidenceEpisode(BaseModel):
    """A period of coherent market behavior.
    Compresses many timer-ticks into a single narrative unit."""
    episode_id: str
    index: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    state_profile: str
    summary: str = ""
    evidence: list[MarketEvidence] = Field(default_factory=list)


class ProtectionOrders(BaseModel):
    stop_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    stop_client_order_id: Optional[str] = None
    tp_client_order_id: Optional[str] = None
    stop_price: float = 0.0
    tp_price: float = 0.0
    working_type: str = "MARK_PRICE"
    price_protect: bool = True
    status: str = "PENDING"
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class VirtualFill(BaseModel):
    """Synthetic execution metadata for SHADOW (virtual) positions.
    Captures the friction parameters applied by VirtualExecutor
    so calibration can reason about entry/exit quality."""
    avg_price: float
    executed_qty: float
    fees: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    fee_bps: float = 0.0


class Position(BaseModel):
    position_id: str = Field(default_factory=_generate_id)
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

    protection_orders: ProtectionOrders = Field(default_factory=ProtectionOrders)

    execution_mode: str = "LIVE"
    origin: str = "NORMAL"
    timeframe: str = "5m"
    exit_price: Optional[float] = None
    exit_fees: Optional[float] = None

    execution_id: Optional[str] = None
    trade_group_id: Optional[str] = None
    candidate_id: Optional[str] = None
    correlation_id: Optional[str] = None
    llm_request_id: Optional[str] = None
    strategy_version: str = "1.0"

    execution_model: str = "fixed_friction_v1"
    execution_model_version: str = "1.0"
    execution_parameters: dict = Field(default_factory=dict)
    active_profile_id: Optional[str] = None
    session_id: Optional[str] = None

    risk_decision: str = ""
    risk_decision_reason: str = ""

    created_by: str = "SCANNER"
    opportunity_source: str = "SCANNER"
    opportunity_id: str = ""

    calibration_model: str = ""
    calibration_version: str = ""
    calibration_data: Optional[dict] = None

    # ── Virtual Execution Metadata ──
    virtual_fill: Optional[VirtualFill] = None

    # ── Evidence Evolution Framework ──
    trade_context: Optional[TradeContext] = None
    initial_evidence: Optional[InitialEvidence] = None
    current_evidence: Optional[MarketEvidence] = None
    evidence_episodes: list[EvidenceEpisode] = Field(default_factory=list)

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", False)
        return super().model_dump(*args, **kwargs)


class CandidateTrade(BaseModel):
    symbol: str
    anchor_symbol: str
    correlation_score: float = 0.0
    signal_strength: float = 0.0
    proposed_side: str
    proposed_quantity: float = 0.0
    opportunity_id: str = ""


class SystemEvent(BaseModel):
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service_name: str
    payload: dict = Field(default_factory=dict)
