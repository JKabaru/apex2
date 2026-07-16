from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Pattern(BaseModel, frozen=True):
    field: str
    value: Any
    frequency: float
    confidence_score: float


class RepresentativeExperience(BaseModel, frozen=True):
    experience_id: str
    similarity_score: float
    why_selected: str


class LiveTrajectory(BaseModel, frozen=True):
    position_id: str
    symbol: str
    side: str
    open_duration_minutes: float
    current_pnl_atr: Optional[float] = None
    episode_count: int = 0
    episodes_summary: list[str] = Field(default_factory=list)
    threat_level: str = "low"


class BiasSummary(BaseModel, frozen=True):
    symbol_distribution: dict[str, int] = Field(default_factory=dict)
    timeframe_distribution: dict[str, int] = Field(default_factory=dict)
    regime_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)


class EvidenceProvenance(BaseModel, frozen=True):
    analysis_version: str = "4.5.0"
    statistics_version: str = "1.0"
    formatter_version: str = "1.0"
    validator_version: str = "1.0"
    source_report_hash: str = ""


class ExperienceEvidence(BaseModel, frozen=True):
    # metadata
    sample_size: int = 0
    evidence_quality: str = "LOW"
    outlier_count: int = 0
    is_sufficient: bool = False

    # outcomes
    win_rate_pct: Optional[float] = None
    median_pnl_atr: Optional[float] = None
    pnl_iqr: Optional[float] = None
    p10_pnl: Optional[float] = None
    p90_pnl: Optional[float] = None

    # timing
    median_duration_bars: Optional[float] = None
    duration_iqr: Optional[float] = None

    # risk
    median_mae_atr: Optional[float] = None
    median_mfe_atr: Optional[float] = None

    # patterns — pure structured data, no strings
    success_patterns: list[Pattern] = Field(default_factory=list)
    failure_patterns: list[Pattern] = Field(default_factory=list)

    # bias & provenance
    bias_summary: BiasSummary = Field(default_factory=BiasSummary)
    representatives: list[RepresentativeExperience] = Field(default_factory=list)
    provenance: EvidenceProvenance = Field(default_factory=EvidenceProvenance)

    # episode / intra-trade trajectory
    avg_episode_count: float = 0.0
    records_with_episodes: int = 0
    total_episodes: int = 0

    # overall confidence as float (0.0-1.0)
    overall_confidence: float = 0.0

    # live intra-trade trajectories (from open positions)
    live_trajectories: list[LiveTrajectory] = Field(default_factory=list)


class PromptContext(BaseModel, frozen=True):
    context_string: str = ""
    section_order: list[str] = Field(default_factory=list)
    template_version: str = "1.0"
    source_evidence_hash: str = ""
    token_count: int = 0
    evidence_tier: int = 4
    evidence_source: str = "COLD_START"
    has_live_data: bool = False
