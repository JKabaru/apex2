from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from src.storage.learning.learning_corpus import LearningCorpus

logger = structlog.get_logger("reflection_engine")


class ReflectionEngine:
    """Periodically reviews past ReasoningEpisodes to generate reflection observations.

    Analysis dimensions:
      • Low-confidence action ratio (BUY/SELL with confidence < min threshold)
      • Critique overturn rate
      • Symbol-level action distribution
      • Symbols with repeated low-quality reasoning
    """

    def __init__(self, corpus: LearningCorpus, min_confidence_threshold: float = 0.3):
        self._corpus = corpus
        self._min_confidence = min_confidence_threshold

    def reflect(self, lookback_minutes: int = 30) -> list[dict[str, Any]]:
        since = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        episodes = self._corpus.get_recent_reasoning_episodes(limit=200, since=since)
        if not episodes:
            return []

        observations: list[dict[str, Any]] = []

        # ── Low-confidence actions ──
        low_conf = [e for e in episodes if e.action in ("BUY", "SELL") and e.confidence < self._min_confidence]
        if low_conf:
            symbols = list({e.symbol for e in low_conf})
            observations.append({
                "category": "reflection_low_confidence",
                "importance": 0.6,
                "data": {
                    "count": len(low_conf),
                    "total_actions": len([e for e in episodes if e.action in ("BUY", "SELL")]),
                    "symbols": symbols,
                    "lookback_minutes": lookback_minutes,
                },
            })
            logger.info(
                "REFLECTION_LOW_CONFIDENCE",
                count=len(low_conf),
                symbols=symbols,
                _force_log=True,
            )

        # ── Critique overturns (via metadata stored on the episode row) ──
        overturns = 0
        for e in episodes:
            meta = e.metadata
            if isinstance(meta, dict) and meta.get("critique_verdict") == "OVERTURN":
                overturns += 1
        if overturns > 0:
            observations.append({
                "category": "reflection_critique_overturns",
                "importance": 0.7,
                "data": {
                    "count": overturns,
                    "total_episodes": len(episodes),
                    "lookback_minutes": lookback_minutes,
                },
            })
            logger.info(
                "REFLECTION_CRITIQUE_OVERTURNS",
                count=overturns,
                total=len(episodes),
                _force_log=True,
            )

        # ── Symbol-level action distribution ──
        from collections import Counter
        symbol_actions: dict[str, Counter] = {}
        for e in episodes:
            symbol_actions.setdefault(e.symbol, Counter())[e.action] += 1
        for sym, counts in symbol_actions.items():
            total = sum(counts.values())
            if total >= 3:
                buy_ratio = counts.get("BUY", 0) / total
                sell_ratio = counts.get("SELL", 0) / total
                hold_ratio = counts.get("HOLD", 0) / total
                if buy_ratio > 0.7 or sell_ratio > 0.7:
                    dominant = "BUY" if buy_ratio > sell_ratio else "SELL"
                    observations.append({
                        "category": "reflection_symbol_bias",
                        "importance": 0.5,
                        "data": {
                            "symbol": sym,
                            "dominant_action": dominant,
                            "ratio": max(buy_ratio, sell_ratio),
                            "total_decisions": total,
                            "lookback_minutes": lookback_minutes,
                        },
                    })
                    logger.info(
                        "REFLECTION_SYMBOL_BIAS",
                        symbol=sym,
                        dominant_action=dominant,
                        ratio=max(buy_ratio, sell_ratio),
                        total=total,
                        _force_log=True,
                    )

        logger.info(
            "REFLECTION_CYCLE_COMPLETE",
            episodes_scanned=len(episodes),
            observations_generated=len(observations),
            _force_log=True,
        )

        return observations
