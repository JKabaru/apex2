from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CalibrationSummary(BaseModel, frozen=True):
    bucket_label: str
    midpoint: float
    low: float
    high: float
    sample_size: int
    wins: int
    win_rate: float
    calibration_error: float
    wilson_ci_low: float
    wilson_ci_high: float


class BiasFinding(BaseModel, frozen=True):
    bias_type: str
    severity: str
    description: str
    metric_value: float
    threshold: float


class ImprovementObservation(BaseModel, frozen=True):
    category: str
    observation: str
    supporting_metric: dict[str, Any]


class RegimeBreakdown(BaseModel, frozen=True):
    source: str
    sample_size: int
    win_rate: float
    avg_confidence: float
    calibration_error: float


class ResearchReport(BaseModel, frozen=True):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    evaluation_version: str
    status: str = "COMPLETE"
    analysis_window: str = ""
    sample_size: int = 0
    skipped_records_count: int = 0
    confidence_calibration: list[CalibrationSummary] = Field(default_factory=list)
    regime_analysis: list[RegimeBreakdown] = Field(default_factory=list)
    risk_analysis: dict[str, Any] = Field(default_factory=dict)
    holding_analysis: dict[str, Any] = Field(default_factory=dict)
    overall_metrics: dict[str, Any] = Field(default_factory=dict)
    bias_findings: list[BiasFinding] = Field(default_factory=list)
    observations: list[ImprovementObservation] = Field(default_factory=list)
