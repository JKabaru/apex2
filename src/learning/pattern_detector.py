from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from src.models.learning.observation import Observation, ObservationCategory
from src.models.learning.pattern import Pattern, PatternCategory

logger = structlog.get_logger("pattern_detector")


class PatternDetector:
    """Detects objective patterns from observation sequences within a timeline.

    Each detection method focuses on one PatternCategory and returns a Pattern
    if the evidence meets its confidence threshold. Patterns are objective
    (what happened) — they do not interpret causation.
    """

    def __init__(self, corpus: Any) -> None:
        self._corpus = corpus

    def detect_all(self, timeline_id: str) -> list[Pattern]:
        """Run every detection strategy and return matched patterns."""
        observations = self._load_observations(timeline_id)
        if not observations:
            return []

        patterns: list[Pattern] = []
        detectors = [
            self._detect_volatility_contraction,
            self._detect_failed_breakout,
            self._detect_price_rejection,
            self._detect_trailing_stop_oscillation,
            self._detect_momentum_exhaustion,
            self._detect_repeated_protection_retry,
        ]
        for detect in detectors:
            try:
                result = detect(observations, timeline_id)
                if result is not None:
                    patterns.append(result)
                    self._corpus.save_pattern(result)
                    logger.info("[PATTERN] Detected", pattern_id=result.pattern_id,
                                 category=result.category.value, timeline_id=timeline_id,
                                 confidence=result.confidence)
            except Exception as e:
                logger.warning("[PATTERN] Detection failed", detector=detect.__name__, error=str(e))

        return patterns

    def _load_observations(self, timeline_id: str) -> list[Observation]:
        links = self._corpus.get_timeline_observations(timeline_id)
        obs_list: list[Observation] = []
        for link in links:
            obs = self._corpus.get_observation(link.observation_id)
            if obs is not None:
                obs_list.append(obs)
        return obs_list

    # ── Detection strategies ──

    def _detect_volatility_contraction(
        self, observations: list[Observation], timeline_id: str,
    ) -> Optional[Pattern]:
        vol = [o for o in observations if o.category == ObservationCategory.VOLATILITY]
        if len(vol) < 3:
            return None

        values = []
        for o in vol:
            v = o.data.get("atr") or o.data.get("volatility_value")
            if v is not None:
                values.append(v)

        if len(values) < 3:
            return None

        recent = values[-3:]
        if all(recent[i] < recent[i + 1] * 0.85 for i in range(len(recent) - 1)):
            confidence = min(0.9, 0.5 + 0.1 * len(vol))
            return Pattern(
                timeline_id=timeline_id,
                category=PatternCategory.VOLATILITY_CONTRACTION,
                description=f"Volatility contraction detected over last {len(vol)} observations",
                observation_ids=[o.observation_id for o in vol],
                start_time=vol[0].timestamp,
                end_time=vol[-1].timestamp,
                confidence=round(confidence, 2),
            )
        return None

    def _detect_failed_breakout(
        self, observations: list[Observation], timeline_id: str,
    ) -> Optional[Pattern]:
        price = [o for o in observations if o.category == ObservationCategory.PRICE_ACTION]
        if len(price) < 2:
            return None

        breakout_obs = [o for o in price if o.data.get("breakout_attempt") is True]
        rejection_obs = [o for o in price if o.data.get("rejected") is True]

        if breakout_obs and rejection_obs:
            first_breakout = breakout_obs[0]
            first_rejection = rejection_obs[0]
            if first_rejection.timestamp > first_breakout.timestamp:
                return Pattern(
                    timeline_id=timeline_id,
                    category=PatternCategory.FAILED_BREAKOUT,
                    description="Price broke out then was rejected",
                    observation_ids=[first_breakout.observation_id, first_rejection.observation_id],
                    start_time=first_breakout.timestamp,
                    end_time=first_rejection.timestamp,
                    confidence=0.70,
                )
        return None

    def _detect_price_rejection(
        self, observations: list[Observation], timeline_id: str,
    ) -> Optional[Pattern]:
        price = [o for o in observations if o.category == ObservationCategory.PRICE_ACTION]
        rejection_obs = [o for o in price if o.data.get("rejected") is True]

        if len(rejection_obs) >= 2:
            delta = rejection_obs[-1].timestamp - rejection_obs[0].timestamp
            if delta <= timedelta(hours=1):
                return Pattern(
                    timeline_id=timeline_id,
                    category=PatternCategory.PRICE_REJECTION,
                    description=f"Price rejected {len(rejection_obs)} times within {delta.total_seconds():.0f}s",
                    observation_ids=[o.observation_id for o in rejection_obs],
                    start_time=rejection_obs[0].timestamp,
                    end_time=rejection_obs[-1].timestamp,
                    confidence=0.65,
                )
        return None

    def _detect_trailing_stop_oscillation(
        self, observations: list[Observation], timeline_id: str,
    ) -> Optional[Pattern]:
        position = [o for o in observations if o.category == ObservationCategory.POSITION]
        stop_obs = [o for o in position if o.data.get("stop_update") is True]

        if len(stop_obs) >= 3:
            return Pattern(
                timeline_id=timeline_id,
                category=PatternCategory.TRAILING_STOP_OSCILLATION,
                description=f"Trailing stop updated {len(stop_obs)} times",
                observation_ids=[o.observation_id for o in stop_obs],
                start_time=stop_obs[0].timestamp,
                end_time=stop_obs[-1].timestamp,
                confidence=0.60,
            )
        return None

    def _detect_momentum_exhaustion(
        self, observations: list[Observation], timeline_id: str,
    ) -> Optional[Pattern]:
        price = [o for o in observations if o.category == ObservationCategory.PRICE_ACTION]
        if len(price) < 5:
            return None

        momentum_vals = []
        for o in price:
            m = o.data.get("momentum") or o.data.get("roc")
            if m is not None:
                momentum_vals.append(m)

        if len(momentum_vals) >= 5:
            first_half = sum(momentum_vals[: len(momentum_vals) // 2]) / (len(momentum_vals) // 2)
            second_half = sum(momentum_vals[len(momentum_vals) // 2:]) / (len(momentum_vals) - len(momentum_vals) // 2)
            if abs(second_half) < abs(first_half) * 0.5 and abs(second_half) < abs(first_half):
                return Pattern(
                    timeline_id=timeline_id,
                    category=PatternCategory.MOMENTUM_EXHAUSTION,
                    description="Momentum declined significantly in second half of position",
                    observation_ids=[o.observation_id for o in price],
                    start_time=price[0].timestamp,
                    end_time=price[-1].timestamp,
                    confidence=round(min(0.85, 0.4 + 0.05 * len(price)), 2),
                )
        return None

    def _detect_repeated_protection_retry(
        self, observations: list[Observation], timeline_id: str,
    ) -> Optional[Pattern]:
        system = [o for o in observations if o.category in (ObservationCategory.SYSTEM, ObservationCategory.RISK)]
        retries = [o for o in system if o.data.get("retry") is True or o.data.get("protection_failure") is True]

        if len(retries) >= 2:
            return Pattern(
                timeline_id=timeline_id,
                category=PatternCategory.REPEATED_PROTECTION_RETRY,
                description=f"Protection retried {len(retries)} times",
                observation_ids=[o.observation_id for o in retries],
                start_time=retries[0].timestamp,
                end_time=retries[-1].timestamp,
                confidence=0.80,
            )
        return None
