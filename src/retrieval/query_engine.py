from __future__ import annotations

from src.retrieval.models import RetrievalQuery, RetrievalRecord
from src.storage.learning.learning_corpus import LearningCorpus


class QueryEngine:
    """Read-only query executor against the LearningCorpus.
    Translates RetrievalQuery into filtered corpus view calls.
    Never builds SQL strings — delegates all persistence to LearningCorpus."""

    def __init__(self, corpus: LearningCorpus):
        self._corpus = corpus

    def update_corpus(self, corpus: LearningCorpus) -> None:
        self._corpus = corpus

    def execute(self, query: RetrievalQuery) -> list[RetrievalRecord]:
        filters: dict = {}

        if query.symbol is not None:
            filters["symbol"] = query.symbol
        if query.timeframe is not None:
            filters["timeframe"] = query.timeframe
        if query.opportunity_id is not None:
            filters["opportunity_id"] = query.opportunity_id
        if query.market_state_hash is not None:
            filters["market_state_hash"] = query.market_state_hash
        if query.trend_regime is not None:
            filters["trend_regime"] = query.trend_regime
        if query.volatility_regime is not None:
            filters["volatility_regime"] = query.volatility_regime
        if query.correlation_regime is not None:
            filters["correlation_regime"] = query.correlation_regime
        if query.min_integrity > 0:
            filters["min_integrity"] = query.min_integrity
        if query.experience_type is not None:
            filters["experience_type"] = query.experience_type

        limit = max(query.max_results * 3, 200)
        return self._corpus.get_corpus_view(limit=limit, filters=filters or None)
