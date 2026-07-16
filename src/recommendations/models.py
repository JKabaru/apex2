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
    workspace_id: str | None = None
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


class AdaptiveParameter(BaseModel, frozen=True):
    """Definition of a single Category B parameter that the system may evolve."""
    parameter_id: str
    config_path: str
    display_name: str
    description: str = ""
    default_value: float
    min_value: float
    max_value: float
    step: float = 0.1
    unit: str = ""
    decay_rate: float = 0.003
    required_evidence_count: int = 20
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AdaptiveVersion(BaseModel, frozen=True):
    """A specific versioned value for an adaptive parameter with full provenance."""
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parameter_id: str
    profile_id: str
    value: float
    previous_value: float
    confidence: float = 0.7
    decay_rate: float = 0.003
    sample_count: int = 0
    required_evidence_count: int = 20
    reason: str = ""
    evidence_ref: str = ""
    source: str = "recommendation"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    superseded_at: Optional[datetime] = None
    status: str = "active"  # active | superseded | rolled_back

    @property
    def effective_confidence(self) -> float:
        if self.superseded_at:
            return 0.0
        days_active = (datetime.utcnow() - self.created_at).total_seconds() / 86400.0
        decayed = self.confidence * ((1.0 - self.decay_rate) ** days_active)
        return max(0.0, decayed)


class AdaptiveDecision(BaseModel, frozen=True):
    """State machine between Recommendation and applied AdaptiveVersion."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recommendation_id: str
    parameter_id: str
    proposed_value: float
    current_value: float
    confidence: float
    sample_count: int
    required_evidence_count: int
    status: str = "pending"  # pending | approved | applied | rejected | superseded
    reason: str = ""
    evidence_summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    decided_at: Optional[datetime] = None


class LearningPolicy(BaseModel, frozen=True):
    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    tier: str = "balanced"  # research | conservative | balanced | aggressive
    is_active: bool = False
    validation_min_score: int = 70
    evidence_min_count: int = 10
    confidence_min: float = 0.4
    noise_max_score: float = 0.3
    auto_approve_candidates: bool = True
    maintenance_interval_hours: int = 6
    confidence_decay_rate: float = 0.005
    duplicate_threshold: float = 0.85
    consolidation_threshold: float = 0.9
    verification_strictness: str = "normal"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryWorkspace(BaseModel, frozen=True):
    """A named independent memory dataset backed by its own DuckDB file."""
    workspace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    db_path: str
    is_active: bool = False
    description: str = ""
    trade_count: int = 0
    size_bytes: int = 0
    profile_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
