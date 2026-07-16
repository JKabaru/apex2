from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import structlog

from src.models.learning.timeline import Timeline, TimelineObservation, TimelineStatus
from src.models.learning.observation import Observation

logger = structlog.get_logger("timeline_manager")


class TimelineManager:
    """Manages timeline lifecycle and observation linking.

    Responsibilities:
      - Resolve which timeline an observation belongs to
      - Create timelines when a new position opens
      - Append observations to timelines (with sequencing)
      - Transition timeline status (OPEN → CLOSED → READY_FOR_ANALYSIS → …)
    """

    def __init__(self, corpus: Any) -> None:
        self._corpus = corpus

    def resolve_timeline_id(self, observation: Observation) -> Optional[str]:
        """Find an existing timeline for this observation, or return None."""
        position_id = (
            observation.data.get("position_id")
            or (observation.context or {}).get("position_id")
        )
        if position_id:
            tl = self._corpus.get_timeline_by_position(position_id)
            if tl is not None:
                return tl.timeline_id

        # Fallback: resolve by symbol for exit and position events.
        # Handles case where reconciler adopted with a different position_id,
        # leaving the original timeline orphaned.
        event = observation.data.get("event", "")
        if event in ("exit_executed", "stop_loss_hit", "take_profit_hit", "time_based_exit"):
            symbol = observation.symbol
            if symbol:
                tl = self._corpus.get_open_timeline_by_symbol(symbol)
                if tl is not None:
                    return tl.timeline_id

        return None

    def get_or_create_timeline(
        self,
        position_id: str,
        symbol: str,
        side: str,
        timeframe: str,
    ) -> str:
        existing = self._corpus.get_timeline_by_position(position_id)
        if existing is not None:
            return existing.timeline_id

        tl = Timeline(
            position_id=position_id,
            symbol=symbol,
            side=side,
            timeframe=timeframe,
            opened_at=datetime.utcnow(),
        )
        tid = self._corpus.save_timeline(tl)
        logger.info("[TIMELINE] Created", timeline_id=tid, position_id=position_id, symbol=symbol)
        return tid

    def append_observation(self, timeline_id: str, observation: Observation) -> int:
        tl = self._corpus.get_timeline(timeline_id)
        if tl is None:
            raise ValueError(f"Timeline {timeline_id} not found")

        seq = (tl.observation_count or 0) + 1
        link = TimelineObservation(
            timeline_id=timeline_id,
            observation_id=observation.observation_id,
            sequence=seq,
            added_at=datetime.utcnow(),
            importance_at_addition=observation.importance,
        )
        self._corpus.save_timeline_observation(link)

        self._corpus._conn.execute(
            "UPDATE timelines SET observation_count = ? WHERE timeline_id = ?",
            [seq, timeline_id],
        )

        logger.debug("[TIMELINE] Appended observation",
                      timeline_id=timeline_id, observation_id=observation.observation_id,
                      sequence=seq, importance=observation.importance)
        return seq

    def close_timeline(self, timeline_id: str) -> bool:
        self._corpus.close_timeline(timeline_id)
        logger.info("[TIMELINE] Closed", timeline_id=timeline_id,
                     status=TimelineStatus.CLOSED.value)
        return True

    def mark_ready_for_analysis(self, timeline_id: str) -> bool:
        self._corpus.update_timeline_status(timeline_id, TimelineStatus.READY_FOR_ANALYSIS)
        logger.info("[TIMELINE] Marked ready for analysis", timeline_id=timeline_id)
        return True

    def mark_analyzed(self, timeline_id: str) -> bool:
        self._corpus.update_timeline_status(timeline_id, TimelineStatus.ANALYZED)
        logger.info("[TIMELINE] Marked analyzed", timeline_id=timeline_id)
        return True

    def mark_archived(self, timeline_id: str) -> bool:
        self._corpus.update_timeline_status(timeline_id, TimelineStatus.ARCHIVED)
        logger.info("[TIMELINE] Archived", timeline_id=timeline_id)
        return True
