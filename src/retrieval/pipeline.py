from __future__ import annotations

import time
from typing import Optional

from src.learning.feature_catalog import FeatureCatalog
from src.retrieval.models import RetrievalQuery, RetrievalRecord, RetrievalContext, SimilarityBreakdown
from src.retrieval.projection import CorpusProjection
from src.retrieval.query_engine import QueryEngine
from src.retrieval.ranking import RankingEngine
from src.retrieval.report import RetrievalReport, build_report
from src.retrieval.similarity import SimilarityEngine
from src.retrieval.weights import SimilarityWeights
from src.storage.learning.learning_corpus import LearningCorpus


class RetrievalPipeline:
    """Orchestrates the deterministic retrieval pipeline.
    Chains: QueryEngine → FeatureProjection → SimilarityEngine → RankingEngine → Report"""

    def __init__(
        self,
        corpus: LearningCorpus,
        feature_catalog: FeatureCatalog,
        weights: SimilarityWeights | None = None,
    ):
        self._query_engine = QueryEngine(corpus)
        self._similarity_engine = SimilarityEngine(feature_catalog, weights=weights)
        self._ranking_engine = RankingEngine()
        self._projector = CorpusProjection()
        self._corpus = corpus

    def update_weights(self, weights: SimilarityWeights) -> None:
        self._similarity_engine.update_weights(weights)

    def update_corpus(self, corpus: LearningCorpus) -> None:
        self._corpus = corpus
        self._query_engine.update_corpus(corpus)

    def retrieve(
        self,
        query: RetrievalQuery,
        context: RetrievalContext | None = None,
    ) -> RetrievalReport:
        start = time.perf_counter()

        query_proj = self._projector.project_query(query)

        candidates = self._query_engine.execute(query)
        candidates_before = len(candidates)

        scores: list[SimilarityBreakdown] = []
        for rec in candidates:
            score = self._similarity_engine.compute_similarity(query_proj, rec)
            scores.append(score)

        ranked = self._ranking_engine.rank(
            candidates, scores, max_results=query.max_results,
        )

        elapsed = (time.perf_counter() - start) * 1000.0

        catalog_version = ""
        meta = self._corpus.get_corpus_metadata()
        if meta is not None:
            catalog_version = meta.feature_catalog_hash

        return build_report(
            query=query,
            candidates_before_filter=candidates_before,
            candidates_after_filter=len(candidates),
            ranked=ranked,
            execution_time_ms=elapsed,
            catalog_version=catalog_version,
            context=context,
        )
