from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from src.models.learning.belief import Belief
from src.storage.learning.learning_corpus import LearningCorpus

logger = structlog.get_logger("belief_evolution")


class BeliefEvolutionEngine:
    """Converts reflection observations into persistent Beliefs.

    Each reflection cycle generates observations (low-confidence actions,
    symbol bias, critique overturns). This engine upserts Beliefs to
    capture those meta-cognitive insights for profile adaptation.
    """

    def __init__(self, corpus: LearningCorpus):
        self._corpus = corpus

    def evolve(self, observations: list[dict[str, Any]]) -> list[Belief]:
        if not observations:
            return []

        evolved: list[Belief] = []
        now = datetime.utcnow()

        for obs in observations:
            category = obs.get("category", "")
            data = obs.get("data", {})
            importance = obs.get("importance", 0.5)

            if category == "reflection_low_confidence":
                beliefs = self._handle_low_confidence(data, importance, now)
                evolved.extend(beliefs)

            elif category == "reflection_critique_overturns":
                beliefs = self._handle_critique_overturns(data, importance, now)
                evolved.extend(beliefs)

            elif category == "reflection_symbol_bias":
                beliefs = self._handle_symbol_bias(data, importance, now)
                evolved.extend(beliefs)

        logger.info(
            "BELIEF_EVOLUTION_CYCLE",
            observations_consumed=len(observations),
            beliefs_created_or_updated=len(evolved),
            _force_log=True,
        )

        return evolved

    def _upsert_belief(self, category: str, symbol: str, statement: str,
                       confidence: float, strength: float, now: datetime,
                       metadata: dict | None = None) -> Belief:
        existing = self._corpus.get_beliefs_by_category(category)
        match = [b for b in existing if b.symbol == symbol and not b.deprecated]

        if match:
            b = match[0]
            new_count = b.observation_count + 1
            new_confidence = min(1.0, b.confidence + 0.05)
            new_strength = max(b.strength, strength)
            self._corpus.update_belief(
                b.belief_id,
                confidence=new_confidence,
                strength=new_strength,
                observation_count=new_count,
                metadata={**(b.metadata or {}), **(metadata or {}),
                          "last_observation": now.isoformat()},
            )
            updated = Belief(
                belief_id=b.belief_id,
                statement=b.statement,
                category=category,
                symbol=symbol,
                confidence=new_confidence,
                strength=new_strength,
                source=b.source,
                observation_count=new_count,
                created_at=b.created_at,
                last_updated=now,
                deprecated=False,
                metadata={**(b.metadata or {}), **(metadata or {}),
                          "last_observation": now.isoformat()},
            )
            logger.info(
                "BELIEF_UPDATED",
                belief_id=b.belief_id,
                category=category,
                symbol=symbol or "*",
                confidence=new_confidence,
                observation_count=new_count,
                _force_log=True,
            )
            return updated
        else:
            belief = Belief(
                statement=statement,
                category=category,
                symbol=symbol,
                confidence=confidence,
                strength=strength,
                source="reflection",
                created_at=now,
                last_updated=now,
                metadata={**(metadata or {}), "first_observation": now.isoformat()},
            )
            self._corpus.save_belief(belief)
            logger.info(
                "BELIEF_CREATED",
                belief_id=belief.belief_id,
                category=category,
                symbol=symbol or "*",
                confidence=confidence,
                _force_log=True,
            )
            return belief

    def _handle_low_confidence(self, data: dict, importance: float, now: datetime) -> list[Belief]:
        count = data.get("count", 0)
        total = data.get("total_actions", 0)
        ratio = count / total if total > 0 else 0
        symbols = data.get("symbols", [])
        statement = f"Agent tends toward low-confidence actions ({count}/{total} = {ratio:.0%})"
        strength = min(1.0, ratio * importance * 2)
        return [self._upsert_belief(
            "low_confidence_tendency", "",
            statement, min(1.0, ratio * 1.5), strength, now,
            metadata={"low_conf_count": count, "total_actions": total},
        )]

    def _handle_critique_overturns(self, data: dict, importance: float, now: datetime) -> list[Belief]:
        count = data.get("count", 0)
        total = data.get("total_episodes", 0)
        ratio = count / total if total > 0 else 0
        statement = f"Critique overturns detected ({count}/{total} = {ratio:.0%} of decisions)"
        strength = min(1.0, ratio * importance * 2)
        return [self._upsert_belief(
            "critique_overturn_rate", "",
            statement, min(1.0, ratio * 1.5), strength, now,
            metadata={"overturn_count": count, "total_episodes": total},
        )]

    def _handle_symbol_bias(self, data: dict, importance: float, now: datetime) -> list[Belief]:
        symbol = data.get("symbol", "")
        dominant = data.get("dominant_action", "")
        ratio = data.get("ratio", 0.0)
        total = data.get("total_decisions", 0)
        statement = f"Symbol bias detected on {symbol}: {ratio:.0%} {dominant} ({total} decisions)"
        strength = min(1.0, (ratio - 0.5) * 2 * importance)
        return [self._upsert_belief(
            "symbol_bias", symbol,
            statement, min(1.0, ratio), strength, now,
            metadata={"dominant_action": dominant, "ratio": ratio, "total_decisions": total},
        )]
