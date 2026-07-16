from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import structlog

from src.core.models import SystemEvent
from src.models.learning.observation import Observation, ObservationCategory, SourceComponent

logger = structlog.get_logger(__name__)

_NON_ROUTE_CATEGORIES = frozenset({"system"})


class ObservationIngestor:
    """Converts OBSERVATION_EMITTED events into the observation pipeline.

    Flow:
      OBSERVATION_EMITTED event
        → parse payload
        → dynamic importance scoring (via ImportanceScorer)
        → save to corpus
        → route to timeline (if importance >= 0.30 and timeline resolved)
        → log

    This is the single entry point for all observations into the memory system.
    Every service emits OBSERVATION_EMITTED — this ingestor handles them uniformly.
    """

    def __init__(self, corpus: Any, scorer: Any, timeline_manager: Any) -> None:
        self._corpus = corpus
        self._scorer = scorer
        self._timeline_manager = timeline_manager
        logger.info("[OBSERVATION] Ingestor initialized")

    async def on_observation_emitted(self, event: SystemEvent) -> None:
        payload = event.payload
        if not payload:
            return

        try:
            symbol = payload.get("symbol", "")
            if not symbol:
                return

            source = self._parse_source(payload.get("source", ""))
            category = self._parse_category(payload.get("category", ""))
            base_importance = payload.get("importance", 0.5)
            raw_data = payload.get("data", {}) or {}
            raw_context = payload.get("context", {}) or {}

            temp_obs = Observation(
                timestamp=event.timestamp,
                source=source,
                category=category,
                importance=base_importance,
                symbol=symbol,
                data=raw_data,
                context=raw_context,
            )

            adjusted = self._scorer.score(temp_obs)

            if adjusted < 0.02:
                logger.debug("[OBSERVATION] Dropped (noise threshold)",
                              importance=round(adjusted, 4), symbol=symbol, category=category.value)
                return

            obs = Observation(
                timestamp=event.timestamp,
                source=source,
                category=category,
                importance=adjusted,
                symbol=symbol,
                data=raw_data,
                context=raw_context,
            )

            self._corpus.save_observation(obs)

            if adjusted >= 0.30 and category not in _NON_ROUTE_CATEGORIES:
                tid = self._timeline_manager.resolve_timeline_id(obs)
                if tid is not None:
                    seq = self._timeline_manager.append_observation(tid, obs)
                    if raw_data.get("event") in ("exit_executed", "stop_loss_hit", "take_profit_hit", "time_based_exit"):
                        self._timeline_manager.close_timeline(tid)
                        self._timeline_manager.mark_ready_for_analysis(tid)
                    logger.info("[OBSERVATION] Routed to timeline",
                                 observation_id=obs.observation_id,
                                 timeline_id=tid, sequence=seq,
                                 importance=round(adjusted, 4))
                elif raw_data.get("event") in ("entry_executed", "trade_executed"):
                    position_id = (raw_context or {}).get("position_id")
                    if position_id:
                        side = raw_data.get("side", "UNKNOWN")
                        timeframe = raw_data.get("timeframe", "5m")
                        tid = self._timeline_manager.get_or_create_timeline(
                            position_id=position_id,
                            symbol=symbol,
                            side=side,
                            timeframe=timeframe,
                        )
                        seq = self._timeline_manager.append_observation(tid, obs)
                        logger.info("[OBSERVATION] Timeline created from entry event",
                                      timeline_id=tid, symbol=symbol,
                                      position_id=position_id, sequence=seq)
                    else:
                        logger.debug("[OBSERVATION] No timeline for entry event (no position_id)",
                                      symbol=symbol, observation_id=obs.observation_id)
            else:
                logger.debug("[OBSERVATION] Not routed (importance below threshold or excluded category)",
                              importance=round(adjusted, 4), category=category.value,
                              symbol=symbol)

            logger.info("[OBSERVATION] Ingested",
                         observation_id=obs.observation_id,
                         category=category.value,
                         source=source.value,
                         importance=round(adjusted, 4),
                         symbol=symbol)

        except Exception as e:
            logger.warning("[OBSERVATION] Ingestion failed", error=str(e))

    @staticmethod
    def _parse_source(raw: str) -> SourceComponent:
        try:
            return SourceComponent(raw)
        except ValueError:
            return SourceComponent.SYSTEM

    @staticmethod
    def _parse_category(raw: str) -> ObservationCategory:
        try:
            return ObservationCategory(raw)
        except ValueError:
            return ObservationCategory.SYSTEM
