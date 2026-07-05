from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceStrength(BaseModel, frozen=True):
    effect_size: float
    magnitude_label: str
    ci_low: float
    ci_high: float


class EvidenceQuality(BaseModel, frozen=True):
    information_weight_score: float
    sample_size: int
    cross_regime_agreement: float
    consistency_score: float
    trustworthiness: str


class SimulationResult(BaseModel, frozen=True):
    intervention_id: str
    simulated_at: datetime = Field(default_factory=datetime.utcnow)
    expected_sharpe_delta: float = 0.0
    expected_win_rate_delta: float = 0.0
    expected_profit_factor_delta: float = 0.0
    expected_trade_frequency_change_pct: float = 0.0
    expected_max_drawdown_change: float = 0.0
    simulated_sample_size: int = 0
    counterfactual_wins: int = 0
    counterfactual_losses: int = 0


class Finding(BaseModel, frozen=True):
    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: str
    description: str
    supporting_metrics: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    severity: str = "MEDIUM"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Intervention(BaseModel, frozen=True):
    intervention_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    finding_id: str
    parameter_id: str
    current_value: Any = None
    recommended_value: Any = None
    reasoning: str = ""
    evidence_strength: EvidenceStrength
    evidence_quality: EvidenceQuality
    sample_size: int = 0
    information_weight_score: float = 0.0
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Recommendation(BaseModel, frozen=True):
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intervention_id: str
    simulation_result: SimulationResult
    why: str = ""
    risk_if_wrong: str = ""
    expected_improvement_pct: float = 0.0
    confidence_tier: str = "LOW"
    evidence_strength: EvidenceStrength
    evidence_quality: EvidenceQuality
    confidence_decay_half_life_days: int = 90
    status: str = "SIMULATED"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConfigurationProfile(BaseModel, frozen=True):
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    base_profile: str = "default"
    parent_profile: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    system_generated: bool = False
    parameter_overrides: dict[str, Any] = Field(default_factory=dict)
    resolved_configuration: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = False
    derived_from_recommendations: list[str] = Field(default_factory=list)
    derived_from_findings: list[str] = Field(default_factory=list)
    activation_reason: str = ""
    created_by: str = "system"


class ActivationRecord(BaseModel, frozen=True):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str
    activated_at: datetime = Field(default_factory=datetime.utcnow)
    deactivated_at: Optional[datetime] = None
    activated_by: str = "system"
