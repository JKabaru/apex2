from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from src.models.learning.observation import Observation, ObservationCategory, SourceComponent
from src.models.learning.observation_aggregate import ObservationAggregate

logger = structlog.get_logger("observation_compressor")


class ObservationCompressor:
    """Compresses similar observations into ObservationAggregates.

    Compression does NOT modify or delete original observations — the
    originals remain independently analyzable. An aggregate summarizes
    N observations within a time window.

    Eligible for compression when:
      - 100+ observations of the same category+symbol exist within 15 min
      - OR observations are explicitly tagged as compressible
    """

    def __init__(self, corpus: Any, window_minutes: int = 15, batch_size: int = 100) -> None:
        self._corpus = corpus
        self._window_minutes = window_minutes
        self._batch_size = batch_size

    def compress_symbol_category(
        self, symbol: str, category: ObservationCategory,
    ) -> Optional[ObservationAggregate]:
        recent = self._corpus.get_recent_observations(
            minutes=self._window_minutes, min_importance=0.0,
        )
        candidates = [
            o for o in recent
            if o.symbol == symbol and o.category == category
        ]

        if len(candidates) < self._batch_size:
            return None

        window_start = candidates[0].timestamp
        window_end = candidates[-1].timestamp
        max_imp = max(o.importance for o in candidates)
        obs_ids = [o.observation_id for o in candidates]

        summary = self._build_summary(candidates)

        agg = ObservationAggregate(
            observation_ids=obs_ids,
            count=len(candidates),
            window_start=window_start,
            window_end=window_end,
            source=candidates[0].source,
            category=category,
            symbol=symbol,
            importance=round(max_imp, 4),
            summary_data=summary,
        )
        aid = self._corpus.save_aggregate(agg)
        logger.info("[COMPRESSOR] Created aggregate", aggregate_id=aid,
                     symbol=symbol, category=category.value, count=len(candidates))
        return agg

    def compress_recent_all(self) -> int:
        symbols = set()
        categories = set()
        recent = self._corpus.get_recent_observations(minutes=self._window_minutes)
        for o in recent:
            symbols.add(o.symbol)
            categories.add(o.category)

        count = 0
        for symbol in symbols:
            for cat in categories:
                try:
                    result = self.compress_symbol_category(symbol, cat)
                    if result is not None:
                        count += 1
                except Exception as e:
                    logger.warning("[COMPRESSOR] Failed", symbol=symbol, category=cat.value, error=str(e))
        return count

    @staticmethod
    def _build_summary(candidates: list[Observation]) -> dict[str, Any]:
        importance_vals = [o.importance for o in candidates]
        return {
            "observation_count": len(candidates),
            "importance_max": round(max(importance_vals), 4),
            "importance_min": round(min(importance_vals), 4),
            "importance_avg": round(sum(importance_vals) / len(importance_vals), 4),
            "sources": list(set(o.source.value for o in candidates)),
            "has_position_context": any("position_id" in (o.data or {}) for o in candidates),
        }
