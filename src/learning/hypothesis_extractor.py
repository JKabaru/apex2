from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from src.models.learning.hypothesis import Hypothesis, HypothesisEvidence, HypothesisStatus
from src.models.learning.pattern import Pattern, PatternCategory

logger = structlog.get_logger("hypothesis_extractor")


class HypothesisExtractor:
    """Extracts hypotheses from patterns in closed timelines.

    A hypothesis is an interpretation of one or more patterns — it answers
    "why did this happen?" rather than "what happened?".

    Multiple hypotheses may coexist for the same timeline, representing
    competing explanations.
    """

    def __init__(self, corpus: Any) -> None:
        self._corpus = corpus

    def extract_all(self, timeline_id: str) -> list[Hypothesis]:
        patterns = self._corpus.get_patterns_by_timeline(timeline_id)
        if not patterns:
            logger.info("[HYPOTHESIS] No patterns to analyze", timeline_id=timeline_id)
            return []

        hypotheses: list[Hypothesis] = []
        generators = [
            self._hypothesize_from_volatility,
            self._hypothesize_from_failed_breakout,
            self._hypothesize_from_rejection,
            self._hypothesize_from_stop_oscillation,
            self._hypothesize_from_momentum,
            self._hypothesize_from_protection_retry,
        ]

        for gen in generators:
            try:
                for hyp in gen(patterns, timeline_id):
                    hypotheses.append(hyp)
                    self._corpus.save_hypothesis(hyp)
                    self._link_evidence(hyp, timeline_id, patterns)
                    logger.info("[HYPOTHESIS] Extracted", hypothesis_id=hyp.hypothesis_id,
                                 statement=hyp.statement[:60], timeline_id=timeline_id)
            except Exception as e:
                logger.warning("[HYPOTHESIS] Extraction failed", generator=gen.__name__, error=str(e))

        return hypotheses

    def _link_evidence(self, hypothesis: Hypothesis, timeline_id: str, patterns: list[Pattern]) -> None:
        for pid in hypothesis.pattern_ids:
            for pat in patterns:
                if pat.pattern_id == pid:
                    for oid in pat.observation_ids:
                        ev = HypothesisEvidence(
                            hypothesis_id=hypothesis.hypothesis_id,
                            timeline_id=timeline_id,
                            observation_id=oid,
                            weight=pat.confidence,
                            supports=True,
                        )
                        self._corpus.save_hypothesis_evidence(ev)

    # ── Interpretations ──

    def _hypothesize_from_volatility(
        self, patterns: list[Pattern], timeline_id: str,
    ) -> list[Hypothesis]:
        vol = [p for p in patterns if p.category == PatternCategory.VOLATILITY_CONTRACTION]
        if not vol:
            return []

        symbols = set()
        for p in vol:
            for oid in p.observation_ids:
                obs = self._corpus.get_observation(oid)
                if obs:
                    symbols.add(obs.symbol)

        results = []
        for symbol in symbols:
            results.append(Hypothesis(
                statement=f"Volatility contraction preceded price movement on {symbol}",
                pattern_ids=[p.pattern_id for p in vol],
                symbol=symbol,
                timeframe="",
                side=None,
                confidence=0.40,
                status=HypothesisStatus.DRAFT,
            ))
        return results

    def _hypothesize_from_failed_breakout(
        self, patterns: list[Pattern], timeline_id: str,
    ) -> list[Hypothesis]:
        failed = [p for p in patterns if p.category == PatternCategory.FAILED_BREAKOUT]
        if not failed:
            return []

        symbols = set()
        for p in failed:
            for oid in p.observation_ids:
                obs = self._corpus.get_observation(oid)
                if obs:
                    symbols.add(obs.symbol)

        return [Hypothesis(
            statement=f"Failed breakout on {s} indicates resistance is holding",
            pattern_ids=[p.pattern_id for p in failed],
            symbol=s,
            timeframe="",
            side=None,
            confidence=0.55,
            status=HypothesisStatus.DRAFT,
        ) for s in symbols]

    def _hypothesize_from_rejection(
        self, patterns: list[Pattern], timeline_id: str,
    ) -> list[Hypothesis]:
        rejected = [p for p in patterns if p.category == PatternCategory.PRICE_REJECTION]
        if not rejected:
            return []

        symbols = set()
        for p in rejected:
            for oid in p.observation_ids:
                obs = self._corpus.get_observation(oid)
                if obs:
                    symbols.add(obs.symbol)

        return [Hypothesis(
            statement=f"Repeated price rejection on {s} suggests strong counter-trend pressure",
            pattern_ids=[p.pattern_id for p in rejected],
            symbol=s,
            timeframe="",
            side=None,
            confidence=0.50,
            status=HypothesisStatus.DRAFT,
        ) for s in symbols]

    def _hypothesize_from_stop_oscillation(
        self, patterns: list[Pattern], timeline_id: str,
    ) -> list[Hypothesis]:
        osc = [p for p in patterns if p.category == PatternCategory.TRAILING_STOP_OSCILLATION]
        if not osc:
            return []

        symbols = set()
        for p in osc:
            for oid in p.observation_ids:
                obs = self._corpus.get_observation(oid)
                if obs:
                    symbols.add(obs.symbol)

        return [Hypothesis(
            statement=f"Frequent trailing stop adjustments on {s} indicate indecision in trend direction",
            pattern_ids=[p.pattern_id for p in osc],
            symbol=s,
            timeframe="",
            side=None,
            confidence=0.35,
            status=HypothesisStatus.DRAFT,
        ) for s in symbols]

    def _hypothesize_from_momentum(
        self, patterns: list[Pattern], timeline_id: str,
    ) -> list[Hypothesis]:
        exhausted = [p for p in patterns if p.category == PatternCategory.MOMENTUM_EXHAUSTION]
        if not exhausted:
            return []

        symbols = set()
        for p in exhausted:
            for oid in p.observation_ids:
                obs = self._corpus.get_observation(oid)
                if obs:
                    symbols.add(obs.symbol)

        return [Hypothesis(
            statement=f"Momentum exhaustion on {s} — trend may be losing conviction",
            pattern_ids=[p.pattern_id for p in exhausted],
            symbol=s,
            timeframe="",
            side=None,
            confidence=0.60,
            status=HypothesisStatus.DRAFT,
        ) for s in symbols]

    def _hypothesize_from_protection_retry(
        self, patterns: list[Pattern], timeline_id: str,
    ) -> list[Hypothesis]:
        retries = [p for p in patterns if p.category == PatternCategory.REPEATED_PROTECTION_RETRY]
        if not retries:
            return []

        symbols = set()
        for p in retries:
            for oid in p.observation_ids:
                obs = self._corpus.get_observation(oid)
                if obs:
                    symbols.add(obs.symbol)

        return [Hypothesis(
            statement=f"Repeated protection retries on {s} — possible systemic execution issue",
            pattern_ids=[p.pattern_id for p in retries],
            symbol=s,
            timeframe="",
            side=None,
            confidence=0.70,
            status=HypothesisStatus.DRAFT,
        ) for s in symbols]
