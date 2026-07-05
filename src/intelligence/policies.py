from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OutlierPolicy(BaseModel, frozen=True):
    method: Literal["IQR", "MAD"] = "IQR"
    multiplier: float = 1.5
    minimum_samples: int = 10
    policy_version: str = "1.0"


class ConfidencePolicy(BaseModel, frozen=True):
    min_sample_size: int = 20
    max_coefficient_of_variation: float = 0.5
    min_avg_integrity: int = 80
    policy_version: str = "1.0"


class AnalysisPolicy(BaseModel, frozen=True):
    minimum_sample_size: int = 5
    minimum_integrity: int = 80
    outlier_policy: OutlierPolicy = Field(default_factory=OutlierPolicy)
    pattern_threshold: float = 0.6
    confidence_policy: ConfidencePolicy = Field(default_factory=ConfidencePolicy)
    representative_count: int = 3
    policy_version: str = "1.0"
