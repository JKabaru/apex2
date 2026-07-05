from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MarketContext(BaseModel, frozen=True):
    symbol: str
    timeframe: str
    current_price: float
    indicators: dict[str, Any] = Field(default_factory=dict)
    trend_regime: str = "UNKNOWN"
    momentum: str = "UNKNOWN"
    volatility_regime: str = "UNKNOWN"
    volume_profile: str = "UNKNOWN"
    correlation_regime: str = "UNKNOWN"
    correlation_score: float = 0.0
    correlations: list[dict[str, Any]] = Field(default_factory=list)


class PortfolioSnapshot(BaseModel, frozen=True):
    live_position_count: int = 0
    live_exposure_usdt: float = 0.0
    total_live_exposure_usdt: float = 0.0
    available_margin: float = 0.0
    max_positions: int = 3
    min_llm_confidence: float = 0.3
    max_live_exposure_usdt: float = 10000.0


class LLMDecision(BaseModel, frozen=True):
    action: Literal["BUY", "SELL", "HOLD", "ABSTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=1000)
    risk_assessment: str = Field(default="", max_length=500)
