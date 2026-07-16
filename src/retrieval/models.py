from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.retrieval_scope import RetrievalScope


class EqualityConstraint(BaseModel, frozen=True):
    field: str
    value: Any


class RangeConstraint(BaseModel, frozen=True):
    field: str
    min: Optional[float] = None
    max: Optional[float] = None


class MembershipConstraint(BaseModel, frozen=True):
    field: str
    values: list[Any]


class RetrievalQuery(BaseModel, frozen=True):
    scope: RetrievalScope = RetrievalScope.EXACT
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    opportunity_id: Optional[str] = None
    market_state_hash: Optional[str] = None
    trend_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    correlation_regime: Optional[str] = None
    experience_type: Optional[str] = None
    min_integrity: int = 0
    max_results: int = 50
    feature_constraints: dict[str, EqualityConstraint | RangeConstraint] = Field(default_factory=dict)
    episode_count: int = 0


class RetrievalContext(BaseModel, frozen=True):
    current_market_regime: Optional[str] = None
    current_volatility: Optional[str] = None
    current_execution_mode: Optional[str] = None
    requested_max_results: int = 50
    minimum_similarity_threshold: float = 0.0
    weight_profile_id: str = "default"


class Explanation(BaseModel, frozen=True):
    group_name: str = ""
    matched_features: list[str] = Field(default_factory=list)
    mismatched_features: list[str] = Field(default_factory=list)
    ignored_features: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0


class SimilarityBreakdown(BaseModel, frozen=True):
    market_score: float = 0.0
    execution_score: float = 0.0
    risk_score: float = 0.0
    outcome_score: float = 0.0
    context_score: float = 0.0
    overall_score: float = 0.0
    explanations: list[Explanation] = Field(default_factory=list)
    group_details: dict[str, Any] = Field(default_factory=dict)


class RetrievalRecord(BaseModel, frozen=True):
    experience_id: str
    position_id: str
    schema_version: str = ""
    pipeline_version: str = ""
    created_at: datetime
    hash: str = ""

    record_source: str = "finalized"

    symbol: str
    timeframe: str
    side: str = ""
    opportunity_id: str = ""
    market_state_hash: str = ""

    trend_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    correlation_regime: Optional[str] = None

    experience_type: str = "final"

    integrity_score: int = 100

    normalized_entry_atr_multiple: Optional[float] = None
    normalized_exit_atr_multiple: Optional[float] = None
    pnl_atr_multiple: Optional[float] = None
    mfe_atr_multiple: Optional[float] = None
    mae_atr_multiple: Optional[float] = None
    entry_rsi_percentile: Optional[float] = None
    entry_volatility_percentile: Optional[float] = None
    holding_duration_minutes: Optional[float] = None
    bars_held: Optional[float] = None
    total_slippage_bps: Optional[float] = None
    total_fees_bps: Optional[float] = None
    realized_rr: Optional[float] = None
    initial_risk_atr_multiple: Optional[float] = None

    evidence_episodes_summary: list[dict] = Field(default_factory=list)
    episode_count: int = 0


class CorpusDiagnostics(BaseModel, frozen=True):
    total_experiences: int = 0
    avg_integrity_score: float = 0.0
    schema_version_distribution: dict[str, int] = Field(default_factory=dict)
    pipeline_version_distribution: dict[str, int] = Field(default_factory=dict)
    catalog_hash_distribution: dict[str, int] = Field(default_factory=dict)
    missing_feature_percentages: dict[str, float] = Field(default_factory=dict)
    regime_distributions: dict[str, dict[str, int]] = Field(default_factory=dict)
    symbol_distribution: dict[str, int] = Field(default_factory=dict)
    timeframe_distribution: dict[str, int] = Field(default_factory=dict)
