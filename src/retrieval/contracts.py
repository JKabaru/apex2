from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.retrieval.models import RetrievalQuery, RetrievalRecord, RetrievalContext, SimilarityBreakdown
from src.retrieval.report import RetrievalReport
from src.retrieval.weights import SimilarityWeights


@runtime_checkable
class IQueryEngine(Protocol):
    def execute(self, query: RetrievalQuery) -> list[RetrievalRecord]:
        ...


@runtime_checkable
class ISimilarityEngine(Protocol):
    def __init__(
        self,
        feature_catalog: Any,
        weights: SimilarityWeights | None = None,
    ) -> None:
        ...

    def compute_similarity(
        self,
        query_proj: dict,
        candidate: RetrievalRecord,
    ) -> SimilarityBreakdown:
        ...


@runtime_checkable
class IRetrievalPipeline(Protocol):
    def retrieve(
        self,
        query: RetrievalQuery,
        context: RetrievalContext | None = None,
    ) -> RetrievalReport:
        ...


@runtime_checkable
class IRankingEngine(Protocol):
    def rank(
        self,
        records: list[RetrievalRecord],
        scores: list[SimilarityBreakdown],
    ) -> list[tuple[RetrievalRecord, SimilarityBreakdown]]:
        ...
