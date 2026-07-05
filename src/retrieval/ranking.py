from __future__ import annotations

from src.retrieval.models import RetrievalRecord, SimilarityBreakdown


class RankingEngine:
    """Deterministic ranking engine.
    Sorting priority (strict):
    1. overall_similarity (descending)
    2. integrity_score (descending)
    3. created_at (descending — recency)
    4. experience_id (ascending — stable tie-breaker)

    No randomness. No sampling. Deterministic for identical inputs."""

    @staticmethod
    def rank(
        records: list[RetrievalRecord],
        scores: list[SimilarityBreakdown],
        max_results: int = 50,
    ) -> list[tuple[RetrievalRecord, SimilarityBreakdown]]:
        if len(records) != len(scores):
            raise ValueError(
                f"records ({len(records)}) and scores ({len(scores)}) must have same length"
            )

        paired = list(zip(records, scores))

        paired.sort(
            key=lambda p: (
                -p[1].overall_score,
                -p[0].integrity_score,
                -p[0].created_at.timestamp(),
                p[0].experience_id,
            ),
        )

        return paired[:max_results]
