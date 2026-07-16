from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import structlog

from src.models.learning.hypothesis import Hypothesis, HypothesisStatus
from src.models.learning.knowledge import Knowledge, KnowledgeConfidence

logger = structlog.get_logger("knowledge_promoter")

# Phrases that indicate semantically similar hypotheses.
_SIMILARITY_GROUPS: list[list[str]] = [
    ["volatility contraction", "vol expansion", "volatility"],
    ["failed breakout", "breakout", "resistance"],
    ["price rejection", "rejection", "rejected"],
    ["momentum exhaustion", "momentum", "exhaustion"],
    ["stop oscillation", "stop", "trailing stop"],
    ["protection", "retry", "execution issue"],
]


class KnowledgePromoter:
    """Promotes hypotheses to knowledge by aggregating cross-timeline evidence.

    Knowledge emerges from hypotheses, not directly from timelines or patterns.
    This component:
      1. Scans hypotheses across timelines
      2. Groups semantically similar hypotheses
      3. Computes contradiction ratio (support / (support + contradiction))
      4. Promotes to appropriate KnowledgeConfidence level
      5. Updates existing knowledge with new evidence
    """

    def __init__(self, corpus: Any) -> None:
        self._corpus = corpus

    def promote_all(self) -> list[Knowledge]:
        """Scan all mature hypotheses and promote cross-cutting knowledge."""
        mature = self._corpus.get_mature_hypotheses(min_confidence=0.0)
        if not mature:
            logger.info("[KNOWLEDGE] No mature hypotheses to promote")
            return []

        existing = self._corpus.get_active_knowledge()
        promoted: list[Knowledge] = []

        groups = self._group_similar(mature)
        for group in groups:
            if len(group) < 2:
                continue

            kn = self._promote_group(group, existing)
            if kn is not None:
                promoted.append(kn)

        return promoted

    def _group_similar(self, hypotheses: list[Hypothesis]) -> list[list[Hypothesis]]:
        """Group hypotheses by semantic similarity using keyword overlap."""
        groups: list[list[Hypothesis]] = [[] for _ in _SIMILARITY_GROUPS]
        unassigned: list[Hypothesis] = []

        for h in hypotheses:
            assigned = False
            lower = h.statement.lower()
            for i, keywords in enumerate(_SIMILARITY_GROUPS):
                if any(kw in lower for kw in keywords):
                    groups[i].append(h)
                    assigned = True
                    break
            if not assigned:
                unassigned.append(h)

        result = [g for g in groups if g]
        if unassigned:
            result.append(unassigned)
        return result

    def _promote_group(
        self, group: list[Hypothesis], existing: list[Knowledge],
    ) -> Knowledge | None:
        supporting = sum(1 for h in group if h.supporting_count >= h.contradicting_count)
        contradicting = len(group) - supporting
        total_evidence = sum(h.evidence_count for h in group)
        total_supporting = sum(h.supporting_count for h in group)
        total_contradicting = sum(h.contradicting_count for h in group)

        if total_evidence == 0:
            return None

        support_ratio = total_supporting / (total_supporting + total_contradicting) if (total_supporting + total_contradicting) > 0 else 0.5
        avg_confidence = sum(h.confidence for h in group) / len(group)
        bayesian_score = self._bayesian_update(avg_confidence, support_ratio, len(group))

        cross_timeline = len(set(h.symbol for h in group))
        symbols = list(set(h.symbol for h in group))
        symbol = symbols[0] if symbols else ""

        if len(group) <= 2:
            level = KnowledgeConfidence.EMERGING
        elif len(group) <= 4 and support_ratio >= 0.6:
            level = KnowledgeConfidence.DEVELOPING
        elif len(group) >= 5 and support_ratio >= 0.75:
            level = KnowledgeConfidence.ESTABLISHED
        else:
            level = KnowledgeConfidence.EMERGING

        representative = max(group, key=lambda h: h.confidence)

        merged_ids = list(set(h.hypothesis_id for h in group))
        for kn in existing:
            if self._matches_existing(kn, group, symbol, merged_ids):
                self._corpus.update_knowledge_confidence(
                    kn.knowledge_id, level,
                    round(bayesian_score, 4),
                    total_supporting, total_contradicting, cross_timeline,
                )
                logger.info("[KNOWLEDGE] Updated", knowledge_id=kn.knowledge_id,
                             level=level.value, score=round(bayesian_score, 4))
                return None

        kn = Knowledge(
            statement=representative.statement,
            hypothesis_ids=merged_ids,
            symbol=symbol,
            timeframe=representative.timeframe,
            confidence=level,
            confidence_score=round(bayesian_score, 4),
            supporting_hypothesis_count=supporting,
            contradicting_hypothesis_count=contradicting,
            cross_timeline_count=cross_timeline,
        )
        kid = self._corpus.save_knowledge(kn)
        logger.info("[KNOWLEDGE] Promoted", knowledge_id=kid, level=level.value,
                     score=round(bayesian_score, 4), hypotheses=len(merged_ids),
                     cross_timeline=cross_timeline)
        return kn

    def _matches_existing(
        self, kn: Knowledge, group: list[Hypothesis], symbol: str,
        merged_ids: list[str],
    ) -> bool:
        if kn.symbol != symbol:
            return False
        overlap = set(kn.hypothesis_ids) & set(merged_ids)
        if overlap:
            return True
        lower_kn = kn.statement.lower()
        for h in group:
            if h.symbol != symbol:
                continue
            keywords = h.statement.lower().split()[:5]
            if any(kw in lower_kn for kw in keywords if len(kw) > 4):
                return True
        return False

    @staticmethod
    def _bayesian_update(
        prior_confidence: float, support_ratio: float, n: int,
    ) -> float:
        if n == 0 or prior_confidence == 0:
            return 0.0
        likelihood = support_ratio
        marginal = prior_confidence * likelihood + (1 - prior_confidence) * (1 - likelihood)
        if marginal == 0:
            return 0.0
        posterior = (prior_confidence * likelihood) / marginal
        return max(0.0, min(1.0, posterior))
