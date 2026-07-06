from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Optional


class ExecutionStatus(Enum):
    CREATED = auto()
    VALIDATED = auto()
    SUBMITTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    PROTECTION_PENDING = auto()
    PROTECTED = auto()
    ACTIVE = auto()
    EXITING = auto()
    CLOSED = auto()


class ValidationOutcomeStatus(Enum):
    PASSED = auto()
    PASSED_WITH_WARNINGS = auto()
    RETRYABLE_FAILURE = auto()
    FATAL_FAILURE = auto()


@dataclass(frozen=True)
class ExecutionPlan:
    symbol: str
    side: str
    order_type: str = "MARKET"
    leverage: int = 5
    reduce_only: bool = False
    cross_margin: bool = True
    time_in_force: str = "GTC"
    protection_strategy: str = "fixed_sl_tp"
    retry_strategy: str = "exponential_backoff"
    expected_slippage_bps: float = 3.0
    expected_fee_bps: float = 4.0
    execution_id: str = ""
    trade_group_id: str = ""
    opportunity_id: str = ""
    llm_confidence: float = 1.0


@dataclass(frozen=True)
class ExecutableTrade:
    symbol: str
    side: str
    trade_side: str
    execution_id: str
    trade_group_id: str
    opportunity_id: str
    plan: ExecutionPlan

    quantity: float
    quantity_str: str
    entry_price: float

    requested_stake: float
    leverage: int

    stop_loss_pct: float
    take_profit_pct: float
    stop_price: float
    tp_price: float

    expected_notional: float
    expected_loss: float
    expected_reward: float
    expected_entry_fee: float
    expected_exit_fee: float
    worst_case_loss: float
    max_allowed_risk: float

    step_size: float = 0.0
    tick_size: float = 0.0
    min_qty: float = 0.0
    max_qty: float = 0.0
    min_notional: float = 0.0

    notional_tolerance: float = 0.0
    risk_tolerance: float = 0.0

    available_balance: float = 0.0
    atr: float = 0.0

    @property
    def trade_side_enum(self) -> str:
        return self.trade_side

    def with_status(self, status: ExecutionStatus) -> ExecutableTrade:
        return self


@dataclass(frozen=True)
class ExecutedTrade:
    trade: ExecutableTrade
    status: ExecutionStatus
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fill_notional: float = 0.0
    fill_commission: float = 0.0
    drift_pct: float = 0.0
    actual_loss: float = 0.0
    protection_data: Optional[dict] = None


@dataclass
class ValidationOutcome:
    status: ValidationOutcomeStatus
    code: str
    message: str


@dataclass
class TradeValidationReport:
    phase: str
    consistency: list[ValidationOutcome] = field(default_factory=list)
    intent: list[ValidationOutcome] = field(default_factory=list)
    exchange: list[ValidationOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        combined = self.consistency + self.intent + self.exchange
        return not any(
            o.status in (ValidationOutcomeStatus.FATAL_FAILURE,)
            for o in combined
        )

    @property
    def has_retryable(self) -> bool:
        combined = self.consistency + self.intent + self.exchange
        return any(o.status == ValidationOutcomeStatus.RETRYABLE_FAILURE for o in combined)

    @property
    def has_warnings(self) -> bool:
        combined = self.consistency + self.intent + self.exchange
        return bool(self.warnings) or any(
            o.status == ValidationOutcomeStatus.PASSED_WITH_WARNINGS
            for o in combined
        )
