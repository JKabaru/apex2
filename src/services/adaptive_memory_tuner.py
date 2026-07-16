from __future__ import annotations

import structlog

from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.weights import SimilarityWeights
from src.storage.learning.learning_corpus import LearningCorpus

logger = structlog.get_logger("adaptive_memory_tuner")


class AdaptiveMemoryTuner:
    """Adjusts retrieval similarity weights based on active beliefs.

    Belief → Weight adjustments:
      • symbol_bias          → reduce context_weight (deprioritize symbol matching)
      • low_confidence_tendency → increase market_weight, reduce outcome_weight
      • critique_overturn_rate  → increase risk_weight, reduce execution_weight

    All adjustments preserve the sum-to-1 constraint by normalizing.
    """

    ADJUSTMENTS: dict[str, dict[str, float]] = {
        "symbol_bias": {
            "context": -0.05,
            "market": +0.03,
            "risk": +0.02,
        },
        "low_confidence_tendency": {
            "market": +0.05,
            "outcome": -0.05,
        },
        "critique_overturn_rate": {
            "risk": +0.05,
            "execution": -0.05,
        },
    }

    def __init__(self, pipeline: RetrievalPipeline, corpus: LearningCorpus):
        self._pipeline = pipeline
        self._corpus = corpus
        self._current_weights = SimilarityWeights()

    def tune(self) -> bool:
        """Read active beliefs and adjust retrieval weights.

        Returns True if weights were changed.
        """
        beliefs = self._corpus.get_active_beliefs()
        if not beliefs:
            return False

        deltas: dict[str, float] = {"market": 0.0, "execution": 0.0, "risk": 0.0, "context": 0.0, "outcome": 0.0}

        for belief in beliefs:
            adj = self.ADJUSTMENTS.get(belief.category)
            if adj is None:
                continue
            if belief.confidence < 0.4:
                continue
            strength = belief.strength
            for group, delta in adj.items():
                deltas[group] += delta * strength

        if all(abs(v) < 1e-6 for v in deltas.values()):
            return False

        base = self._current_weights.to_dict()
        new_vals = {
            "market": max(0.0, base["market"] + deltas["market"]),
            "execution": max(0.0, base["execution"] + deltas["execution"]),
            "risk": max(0.0, base["risk"] + deltas["risk"]),
            "context": max(0.0, base["context"] + deltas["context"]),
            "outcome": max(0.0, base["outcome"] + deltas["outcome"]),
        }

        total = sum(new_vals.values())
        if total > 1e-6:
            for k in new_vals:
                new_vals[k] /= total

        try:
            new_weights = SimilarityWeights(
                market_weight=round(new_vals["market"], 4),
                execution_weight=round(new_vals["execution"], 4),
                risk_weight=round(new_vals["risk"], 4),
                context_weight=round(new_vals["context"], 4),
                outcome_weight=round(new_vals["outcome"], 4),
            )
        except ValueError as e:
            logger.warning("MEMORY_TUNE_INVALID_WEIGHTS", error=str(e))
            return False

        old = self._current_weights.to_dict()
        self._pipeline.update_weights(new_weights)
        self._current_weights = new_weights

        logger.info(
            "MEMORY_TUNE_APPLIED",
            old_weights=old,
            new_weights=new_vals,
            deltas=deltas,
            _force_log=True,
        )
        return True
