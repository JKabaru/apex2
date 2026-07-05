from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class SimilarityWeights(BaseModel, frozen=True):
    market_weight: float = 0.35
    execution_weight: float = 0.10
    risk_weight: float = 0.20
    context_weight: float = 0.25
    outcome_weight: float = 0.10

    @field_validator(
        "market_weight", "execution_weight", "risk_weight",
        "context_weight", "outcome_weight", mode="before",
    )
    @classmethod
    def _ensure_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"Weight must be non-negative, got {v}")
        return v

    @model_validator(mode="after")
    def _sum_to_one(self) -> SimilarityWeights:
        total = (
            self.market_weight
            + self.execution_weight
            + self.risk_weight
            + self.context_weight
            + self.outcome_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self

    def to_dict(self) -> dict[str, float]:
        return {
            "market": self.market_weight,
            "execution": self.execution_weight,
            "risk": self.risk_weight,
            "context": self.context_weight,
            "outcome": self.outcome_weight,
        }


class WeightProfileRegistry:
    _profiles: dict[str, SimilarityWeights] = {}

    DEFAULT = SimilarityWeights()
    MOMENTUM = SimilarityWeights(
        market_weight=0.50,
        execution_weight=0.05,
        risk_weight=0.15,
        context_weight=0.20,
        outcome_weight=0.10,
    )
    MEAN_REVERSION = SimilarityWeights(
        market_weight=0.25,
        execution_weight=0.05,
        risk_weight=0.25,
        context_weight=0.30,
        outcome_weight=0.15,
    )

    @classmethod
    def register(cls, profile_id: str, weights: SimilarityWeights) -> None:
        cls._profiles[profile_id] = weights

    @classmethod
    def get(cls, profile_id: str) -> SimilarityWeights:
        if profile_id in cls._profiles:
            return cls._profiles[profile_id]
        name = profile_id.upper().replace("-", "_")
        if hasattr(cls, name):
            return getattr(cls, name)
        return cls.DEFAULT
