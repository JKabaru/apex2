from __future__ import annotations

from collections import Counter
from typing import Callable

from src.intelligence.models import Pattern
from src.intelligence.policies import AnalysisPolicy
from src.retrieval.models import RetrievalRecord


class PatternExtractor:
    """Generic pattern extraction driven by lambda predicates.

    Never hardcodes "success" or "failure" — the caller provides
    the predicate that determines which records belong to each group.
    """

    @staticmethod
    def extract_patterns(
        records: list[RetrievalRecord],
        target_field: str,
        predicate: Callable[[RetrievalRecord], bool],
        policy: AnalysisPolicy,
    ) -> list[Pattern]:
        """Extract structured patterns from categorical fields.

        Args:
            records: Full set of retrieval records.
            target_field: Name of the categorical field to analyze
                         (e.g. 'trend_regime', 'volatility_regime').
            predicate: Lambda that returns True for records in the group.
            policy: Controls minimum sample, frequency threshold.

        Returns:
            List of Pattern objects with float confidence scores.
        """
        matching = [r for r in records if predicate(r)]
        total = len(matching)

        if total < policy.minimum_sample_size:
            return []

        value_counts: Counter[object] = Counter()
        for r in matching:
            val = getattr(r, target_field, None)
            if val is not None:
                value_counts[val] += 1

        patterns: list[Pattern] = []
        for value, count in value_counts.items():
            frequency = count / total
            if frequency >= policy.pattern_threshold:
                confidence = _compute_pattern_confidence(total, count, policy)
                patterns.append(Pattern(
                    field=target_field,
                    value=value,
                    frequency=round(frequency, 4),
                    confidence_score=round(confidence, 4),
                ))

        return patterns


def _compute_pattern_confidence(
    total: int,
    count: int,
    policy: AnalysisPolicy,
) -> float:
    """Confidence increases with sample size and absolute count."""
    cp = policy.confidence_policy
    size_factor = min(1.0, total / cp.min_sample_size)
    count_factor = min(1.0, count / max(total * policy.pattern_threshold, 1))
    return size_factor * 0.5 + count_factor * 0.5
