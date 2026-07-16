from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field


def _generate_id() -> str:
    return str(uuid.uuid4())


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for hashing.
    - Sorted keys for stable field ordering across Python versions
    - Stable ISO format (Z suffix, never +00:00)
    - UTF-8 encoding
    - No whitespace between tokens"""
    def _serialize(v: Any) -> Any:
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%dT%H:%M:%SZ")
        return v

    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        default=_serialize,
        separators=(",", ":"),
    )


def _compute_hash(*components: BaseModel) -> str:
    raw = "".join(
        _canonical_json(c.model_dump(mode="json")) for c in components
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class ConfigurationEntry(BaseModel, frozen=True):
    """A single configuration parameter value with its provenance."""
    parameter_id: str
    value: Any
    source: str = "unknown"
    loaded_at: str = ""
    config_version: str = ""


class ConfigurationSnapshot(BaseModel, frozen=True):
    """Snapshot of all runtime configuration values at the time the trade was extracted."""
    items: list[ConfigurationEntry] = Field(default_factory=list)


class OpportunityIdentity(BaseModel, frozen=True):
    """Identifies the market opportunity that generated this trade.
    The UUID is a unique identifier. The market_state_hash groups
    similar opportunities by market regime."""
    opportunity_id: str
    market_state_hash: str = ""
    market_state_schema_version: str = "1.0"
    scanner_version: str = ""
    strategy_name: str = ""
    strategy_version: str = ""
    discovered_at: datetime
    anchor_symbol: str
    symbol: str
    timeframe: str
    thesis_hash: str = ""


class ManifestProvenance(BaseModel, frozen=True):
    """Software identity that produced this manifest.
    Enables exact reproduction of the pipeline environment."""
    position_schema_version: str = ""
    event_schema_version: str = ""
    pipeline_version: str = ""
    git_commit: str = ""
    build_id: str = ""
    application_version: str = ""


class PositionSnapshot(BaseModel, frozen=True):
    """Frozen boundary between the trading engine and the learning pipeline.
    Created once at position close.

    TRUST BOUNDARY: No learning stage may query runtime services
    after this point. All data required for learning must be present
    in this snapshot. NEVER modified after creation."""
    schema_version: str = "1.0"
    frozen_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Core identity ──
    position_id: str
    symbol: str
    side: str
    timeframe: str

    # ── Execution context ──
    execution_mode: str
    origin: str
    quantity: float
    avg_fill_price: float
    fees: float
    exit_price: Optional[float]
    exit_fees: Optional[float]
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    exit_reason: Optional[str]
    slippage_bps: Optional[float]
    spread_bps: Optional[float]

    # ── Performance ──
    highest_unrealized_profit: float
    maximum_drawdown: float

    # ── Context ──
    anchor_symbol: str
    entry_thesis: str
    correlation_score: float
    initial_stop_loss: float
    initial_take_profit: float
    execution_parameters: dict = Field(default_factory=dict)

    # ── Evidence ──
    entry_atr: Optional[float]
    entry_rsi: Optional[float]
    exit_atr: Optional[float]
    exit_rsi: Optional[float]
    trend_regime: Optional[str]
    volatility_regime: Optional[str]
    correlation_regime: Optional[str]
    evidence_episodes: list[dict] = Field(default_factory=list)

    # ── Calibration ──
    calibration_data: Optional[dict] = None

    # ── Opportunity ──
    opportunity_id: str = ""

    # ── Experience type ──
    experience_type: Literal["interim", "final"] = "final"

    # ── Traceability ──
    active_profile_id: Optional[str] = None
    session_id: Optional[str] = None

    # ── Versioning ──
    pipeline_version: str = "1.0"

    @staticmethod
    def from_position(position) -> PositionSnapshot:
        import copy
        from src.core.models import (
            TradeContext as RuntimeTradeContext,
            VirtualFill as RuntimeVirtualFill,
            EvidenceEpisode as RuntimeEvidenceEpisode,
        )

        evidence = []
        if hasattr(position, "evidence_episodes") and position.evidence_episodes:
            for ep in position.evidence_episodes:
                episode_data = ep.model_dump() if isinstance(ep, RuntimeEvidenceEpisode) else ep
                evidence.append({
                    "episode_id": episode_data.get("episode_id", ""),
                    "index": episode_data.get("index", 0),
                    "state_profile": episode_data.get("state_profile", ""),
                    "started_at": (
                        episode_data["started_at"].isoformat()
                        if isinstance(episode_data.get("started_at"), datetime)
                        else str(episode_data.get("started_at", ""))
                    ),
                    "ended_at": (
                        episode_data["ended_at"].isoformat()
                        if isinstance(episode_data.get("ended_at"), datetime)
                        else (str(episode_data["ended_at"]) if episode_data.get("ended_at") else None)
                    ),
                })

        slippage = None
        spread = None
        if position.virtual_fill:
            vf = position.virtual_fill
            if isinstance(vf, RuntimeVirtualFill):
                slippage = vf.slippage_bps
                spread = vf.spread_bps
            elif isinstance(vf, dict):
                slippage = vf.get("slippage_bps")
                spread = vf.get("spread_bps")
        if slippage is None:
            ep = position.execution_parameters or {}
            slippage = ep.get("slippage_bps")
            spread = ep.get("spread_bps")

        ie = position.initial_evidence
        ce = position.current_evidence

        return PositionSnapshot(
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            timeframe=getattr(position, "timeframe", "5m"),
            execution_mode=position.execution_mode,
            origin=position.origin,
            quantity=position.quantity,
            avg_fill_price=position.avg_fill_price,
            fees=position.fees,
            exit_price=getattr(position, "exit_price", None),
            exit_fees=getattr(position, "exit_fees", None),
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=position.exit_timestamp,
            exit_reason=position.exit_reason,
            slippage_bps=slippage,
            spread_bps=spread,
            highest_unrealized_profit=position.highest_unrealized_profit,
            maximum_drawdown=position.maximum_drawdown,
            anchor_symbol=position.anchor_symbol,
            entry_thesis=position.entry_thesis,
            correlation_score=position.correlation_score,
            initial_stop_loss=position.initial_stop_loss,
            initial_take_profit=position.initial_take_profit,
            execution_parameters=copy.deepcopy(position.execution_parameters or {}),
            entry_atr=ie.atr if ie else None,
            entry_rsi=ie.rsi if ie else None,
            exit_atr=ce.atr if ce else None,
            exit_rsi=ce.rsi if ce else None,
            trend_regime=ie.trend_regime if ie else None,
            volatility_regime=ie.volatility_regime if ie else None,
            correlation_regime=ie.correlation_regime if ie else None,
            evidence_episodes=evidence,
            calibration_data=(
                copy.deepcopy(position.calibration_data)
                if position.calibration_data else None
            ),
            opportunity_id=getattr(position, "opportunity_id", ""),
            active_profile_id=getattr(position, "active_profile_id", None),
            session_id=getattr(position, "session_id", None),
        )

    @staticmethod
    def from_position_interim(position) -> PositionSnapshot:
        import copy
        from src.core.models import (
            TradeContext as RuntimeTradeContext,
            VirtualFill as RuntimeVirtualFill,
            EvidenceEpisode as RuntimeEvidenceEpisode,
        )

        evidence = []
        if hasattr(position, "evidence_episodes") and position.evidence_episodes:
            for ep in position.evidence_episodes:
                episode_data = ep.model_dump() if isinstance(ep, RuntimeEvidenceEpisode) else ep
                evidence.append({
                    "episode_id": episode_data.get("episode_id", ""),
                    "index": episode_data.get("index", 0),
                    "state_profile": episode_data.get("state_profile", ""),
                    "started_at": (
                        episode_data["started_at"].isoformat()
                        if isinstance(episode_data.get("started_at"), datetime)
                        else str(episode_data.get("started_at", ""))
                    ),
                    "ended_at": (
                        episode_data["ended_at"].isoformat()
                        if isinstance(episode_data.get("ended_at"), datetime)
                        else (str(episode_data["ended_at"]) if episode_data.get("ended_at") else None)
                    ),
                })

        slippage = None
        spread = None
        if position.virtual_fill:
            vf = position.virtual_fill
            if isinstance(vf, RuntimeVirtualFill):
                slippage = vf.slippage_bps
                spread = vf.spread_bps
            elif isinstance(vf, dict):
                slippage = vf.get("slippage_bps")
                spread = vf.get("spread_bps")
        if slippage is None:
            ep = position.execution_parameters or {}
            slippage = ep.get("slippage_bps")
            spread = ep.get("spread_bps")

        ie = position.initial_evidence
        ce = position.current_evidence

        return PositionSnapshot(
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            timeframe=getattr(position, "timeframe", "5m"),
            execution_mode=position.execution_mode,
            origin=position.origin,
            quantity=position.quantity,
            avg_fill_price=position.avg_fill_price,
            fees=position.fees,
            exit_price=None,
            exit_fees=None,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=None,
            exit_reason=None,
            slippage_bps=slippage,
            spread_bps=spread,
            highest_unrealized_profit=position.highest_unrealized_profit,
            maximum_drawdown=position.maximum_drawdown,
            anchor_symbol=position.anchor_symbol,
            entry_thesis=position.entry_thesis,
            correlation_score=position.correlation_score,
            initial_stop_loss=position.initial_stop_loss,
            initial_take_profit=position.initial_take_profit,
            execution_parameters=copy.deepcopy(position.execution_parameters or {}),
            entry_atr=ie.atr if ie else None,
            entry_rsi=ie.rsi if ie else None,
            exit_atr=ce.atr if ce else None,
            exit_rsi=ce.rsi if ce else None,
            trend_regime=ie.trend_regime if ie else None,
            volatility_regime=ie.volatility_regime if ie else None,
            correlation_regime=ie.correlation_regime if ie else None,
            evidence_episodes=evidence,
            calibration_data=None,
            opportunity_id=getattr(position, "opportunity_id", ""),
            active_profile_id=getattr(position, "active_profile_id", None),
            session_id=getattr(position, "session_id", None),
            experience_type="interim",
        )


class LearningExperience(BaseModel, frozen=True):
    """Pure trade facts. Contains only what happened — no metadata
    about how it should be interpreted."""
    experience_id: str = Field(default_factory=_generate_id)
    position_id: str
    schema_version: str = "1.0"
    pipeline_version: str = "1.0"
    extraction_version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Experience type ──
    experience_type: Literal["interim", "final"] = "final"

    # ── Pipeline-transformable data ──
    symbol: str
    side: str = ""
    timeframe: str
    entry_price: float
    exit_price: Optional[float]
    fees: float
    exit_fees: Optional[float]
    highest_unrealized_profit: float
    maximum_drawdown: float
    slippage_bps: Optional[float]
    spread_bps: Optional[float]

    entry_atr: Optional[float]
    entry_rsi: Optional[float]
    exit_atr: Optional[float]
    exit_rsi: Optional[float]
    trend_regime: Optional[str]
    volatility_regime: Optional[str]
    correlation_regime: Optional[str]
    calibration_data: Optional[dict] = None

    evidence_episodes_summary: list[dict] = Field(default_factory=list)
    episode_count: int = 0

    # ── Identity cross-reference (scalar only — full object in manifest) ──
    opportunity_id: str = ""


class ValidationReport(BaseModel, frozen=True):
    """Transparent listing of every verification check performed."""
    schema_version: str = "1.0"
    validator_version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    verified_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    computed_fields: list[str] = Field(default_factory=list)
    schema_errors: list[str] = Field(default_factory=list)
    ordering_errors: list[str] = Field(default_factory=list)
    evidence_notes: list[str] = Field(default_factory=list)

    @property
    def integrity_score(self) -> int:
        score = 100
        score -= len(self.missing_fields) * 5
        score -= len(self.schema_errors) * 10
        score -= len(self.ordering_errors) * 8
        score -= len(self.evidence_notes) * 3
        return max(0, min(100, score))

    @property
    def field_quality(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for f in self.verified_fields:
            result[f] = "VERIFIED"
        for f in self.missing_fields:
            result[f] = "MISSING"
        for f in self.computed_fields:
            result[f] = "COMPUTED"
        return result


class NormalizedMetrics(BaseModel, frozen=True):
    """Normalized values computed from the LearningExperience.
    Raw facts remain untouched in LearningExperience."""
    schema_version: str = "1.0"
    normalizer_version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Market Normalization ──
    normalized_entry_atr_multiple: Optional[float] = None
    normalized_exit_atr_multiple: Optional[float] = None
    pnl_atr_multiple: Optional[float] = None
    mfe_atr_multiple: Optional[float] = None
    mae_atr_multiple: Optional[float] = None
    entry_rsi_percentile: Optional[float] = None
    entry_volatility_percentile: Optional[float] = None
    holding_duration_minutes: Optional[float] = None
    bars_held: Optional[float] = None

    # ── Execution Normalization ──
    total_slippage_bps: Optional[float] = None
    total_fees_bps: Optional[float] = None
    realized_rr: Optional[float] = None
    initial_risk_atr_multiple: Optional[float] = None


class DuplicateResult(BaseModel, frozen=True):
    is_duplicate: bool = False
    matched_experience_id: Optional[str] = None
    match_score: float = 0.0
    matched_fields: list[str] = []
    action: str = "none"  # skip | merge | notified

class NoiseAssessment(BaseModel, frozen=True):
    is_noise: bool = False
    noise_score: float = 0.0
    noise_factors: list[str] = []
    reason: str = ""

class ConfidenceScore(BaseModel, frozen=True):
    score: float = 0.0
    evaluation_quality: float = 0.0
    execution_integrity: float = 0.0
    evidence_strength: float = 0.0
    data_completeness: float = 0.0
    consistency: float = 0.0
    verifier_version: str = "1.0"

class PersistenceVerification(BaseModel, frozen=True):
    verified: bool = True
    hash_matches: bool = True
    read_back_ok: bool = True
    index_ok: bool = True
    workspace_ok: bool = True
    visibility_ok: bool = True
    error: str = ""

class MaintenanceReport(BaseModel, frozen=True):
    experiences_scanned: int = 0
    duplicates_merged: int = 0
    confidence_updates: int = 0
    database_optimized: bool = False
    integrity_verified: bool = False
    workspace_size_bytes: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = []

class MemoryHealth(BaseModel, frozen=True):
    experience_count: int = 0
    pending_candidates: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    workspace_size_bytes: int = 0
    database_size_bytes: int = 0
    integrity_state: str = "unknown"
    verification_state: str = "unknown"
    last_maintenance: Optional[str] = None
    last_save: Optional[str] = None

class LearningManifest(BaseModel, frozen=True):
    """The ONLY persisted artifact. Wraps every sub-artifact produced by the pipeline.
    Like a Git commit for a trade — self-describing and content-addressable."""
    experience_id: str
    position_id: str
    schema_version: str = "2.0"
    pipeline_version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    hash: str = ""

    # ── Experience type ──
    experience_type: Literal["interim", "final"] = "final"

    # ── Sub-artifacts ──
    learning_experience: LearningExperience
    validation_report: ValidationReport
    normalized_metrics: NormalizedMetrics

    # ── Stage 2: Metadata ──
    feature_catalog_version: str = ""
    feature_catalog_hash: str = ""
    configuration_snapshot: ConfigurationSnapshot = Field(default_factory=ConfigurationSnapshot)
    active_profile_id: Optional[str] = None
    session_id: Optional[str] = None
    provenance_version: str = ""
    opportunity_identity: Optional[OpportunityIdentity] = None
    manifest_provenance: ManifestProvenance = Field(default_factory=ManifestProvenance)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.hash:
            artifacts = [
                self.learning_experience,
                self.validation_report,
                self.normalized_metrics,
                self.configuration_snapshot,
            ]
            if self.opportunity_identity is not None:
                artifacts.append(self.opportunity_identity)
            artifacts.append(self.manifest_provenance)
            raw = "".join(
                _canonical_json(a.model_dump(mode="json")) for a in artifacts
            )
            object.__setattr__(self, "hash", hashlib.sha256(raw.encode()).hexdigest())
