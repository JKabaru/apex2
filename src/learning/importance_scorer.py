from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import structlog

from src.models.learning.observation import Observation, ObservationCategory

logger = structlog.get_logger("importance_scorer")


class ImportanceScorer:
    """Dynamic importance scoring.

    Importance is computed as:

        score = base_importance * novelty * rarity * consequence

    clamped to [0.0, 1.0].

    Each factor is derived from corpus history so that the same observation
    type may carry different weight depending on context (e.g. ENTRY_EXECUTED
    is normally 0.60 but 0.95 during a flash crash).
    """

    def __init__(self, corpus: Any) -> None:
        self._corpus = corpus

    def score(self, observation: Observation) -> float:
        """Compute the dynamic importance of an observation."""
        if observation.importance < 0.02:
            return observation.importance

        n = self._compute_novelty(observation)
        r = self._compute_rarity(observation)
        c = self._compute_consequence(observation)

        raw = observation.importance * n * r * c
        return max(0.0, min(1.0, raw))

    def _compute_novelty(self, observation: Observation) -> float:
        """Novelty based on how many similar observations exist in the last 15 min.

        First observation of its kind → 1.0 (maximum novelty).
        Each subsequent similar observation reduces novelty.
        """
        recent = self._corpus.get_recent_observations(minutes=15, min_importance=0.0)
        similar = [
            o for o in recent
            if o.category == observation.category and o.symbol == observation.symbol
        ]
        if not similar:
            return 1.0
        return max(0.3, 1.0 - len(similar) * 0.05)

    def _compute_rarity(self, observation: Observation) -> float:
        """Rarity based on total count of this category + symbol pair."""
        all_similar = self._corpus.query_observations(
            symbol=observation.symbol,
            category=observation.category.value,
        )
        if not all_similar:
            return 1.0
        return max(0.3, 1.0 - len(all_similar) * 0.01)

    def _compute_consequence(self, observation: Observation) -> float:
        """Consequence based on context (market conditions, category severity)."""
        ctx = observation.context or {}
        vol = ctx.get("volatility", "normal")

        if vol == "extreme":
            return 1.20
        if vol == "high":
            return 1.10

        if observation.category in (ObservationCategory.RISK, ObservationCategory.EXECUTION):
            return 1.15

        return 1.0
