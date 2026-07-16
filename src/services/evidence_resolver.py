from __future__ import annotations

import time
from typing import Optional

import structlog

from src.core.models import CandidateTrade
from src.intelligence.models import PromptContext
from src.intelligence.pipeline import ExperienceIntelligencePipeline
from src.models.retrieval_scope import RetrievalScope
from src.retrieval.models import RetrievalContext, RetrievalQuery
from src.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger("evidence_resolver")

TIER_CACHE_TTL: dict[int, float] = {
    1: 300.0,
    2: 300.0,
    3: 60.0,
}

TIER_SOURCE: dict[int, str] = {
    1: "EXACT",
    2: "ANCHOR",
    3: "REGIME",
}


class EvidenceResolver:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        intel_pipeline: ExperienceIntelligencePipeline,
    ):
        self._retrieval = retrieval_pipeline
        self._intel = intel_pipeline
        self._cache: dict[str, tuple[PromptContext, float]] = {}

    def resolve(self, candidate: CandidateTrade) -> PromptContext:
        tiers: list[tuple[int, str, Optional[str]]] = [
            (1, "EXACT", candidate.symbol),
            (2, "ANCHOR", candidate.anchor_symbol),
            (3, "REGIME", None),
        ]

        for tier, source, lookup_key in tiers:
            ctx = self._try_tier(tier, source, candidate, lookup_key)
            if ctx is not None and ctx.context_string:
                return ctx

        logger.info(
            "No historical evidence found at any tier, using cold start",
            symbol=candidate.symbol,
        )
        return PromptContext(
            evidence_tier=4,
            evidence_source="COLD_START",
        )

    def _try_tier(
        self,
        tier: int,
        source: str,
        candidate: CandidateTrade,
        lookup_key: Optional[str],
    ) -> Optional[PromptContext]:
        cache_key = f"{tier}:{lookup_key or 'global'}"

        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        symbol = (
            candidate.symbol
            if source == "EXACT"
            else candidate.anchor_symbol if source == "ANCHOR"
            else None
        )

        query = RetrievalQuery(
            scope=RetrievalScope(source),
            symbol=symbol,
            max_results=50,
        )

        context = RetrievalContext(
            requested_max_results=50,
            minimum_similarity_threshold=0.0,
        )

        report = self._retrieval.retrieve(query, context)
        if not report.results:
            logger.debug(
                "No results for tier",
                tier=tier,
                source=source,
                symbol=symbol,
            )
            return None

        evidence = self._intel.process(report)
        ctx = self._intel.generate_prompt_context(evidence)

        ctx = PromptContext(
            context_string=ctx.context_string,
            section_order=ctx.section_order,
            template_version=ctx.template_version,
            source_evidence_hash=ctx.source_evidence_hash,
            token_count=ctx.token_count,
            evidence_tier=tier,
            evidence_source=source,
            has_live_data=ctx.has_live_data,
        )

        if not ctx.has_live_data:
            self._set_cache(cache_key, ctx)

        logger.info(
            "EVIDENCE_FEEDBACK",
            tier=tier,
            source=source,
            symbol=candidate.symbol,
            sample_size=evidence.sample_size,
            is_sufficient=evidence.is_sufficient,
            token_count=ctx.token_count,
            has_live_data=ctx.has_live_data,
            live_trajectory_count=len(evidence.live_trajectories),
            win_rate=evidence.win_rate if hasattr(evidence, "win_rate") else None,
            total_pnl_atr=evidence.total_pnl_atr if hasattr(evidence, "total_pnl_atr") else None,
            _force_log=True,
        )
        return ctx

    def _check_cache(self, key: str) -> Optional[PromptContext]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ctx, ts = entry
        tier = ctx.evidence_tier
        ttl = TIER_CACHE_TTL.get(tier, 60.0)
        if time.monotonic() - ts > ttl:
            del self._cache[key]
            return None
        return ctx

    def _set_cache(self, key: str, ctx: PromptContext) -> None:
        self._cache[key] = (ctx, time.monotonic())
