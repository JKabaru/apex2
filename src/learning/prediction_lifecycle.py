from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog

from src.models.learning.observation import Observation, ObservationCategory
from src.models.learning.timeline import TimelineStatus

logger = structlog.get_logger("prediction_lifecycle")


class PredictionState(str, Enum):
    CREATED = "created"
    OBSERVED = "observed"
    STRENGTHENED = "strengthened"
    WEAKENED = "weakened"
    THREATENED = "threatened"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    COMPLETED = "completed"


_VALID_TRANSITIONS: dict[PredictionState, set[PredictionState]] = {
    PredictionState.CREATED: {PredictionState.OBSERVED},
    PredictionState.OBSERVED: {PredictionState.STRENGTHENED, PredictionState.WEAKENED, PredictionState.THREATENED, PredictionState.CONFIRMED, PredictionState.INVALIDATED},
    PredictionState.STRENGTHENED: {PredictionState.STRENGTHENED, PredictionState.WEAKENED, PredictionState.THREATENED, PredictionState.CONFIRMED, PredictionState.INVALIDATED},
    PredictionState.WEAKENED: {PredictionState.STRENGTHENED, PredictionState.WEAKENED, PredictionState.THREATENED, PredictionState.INVALIDATED},
    PredictionState.THREATENED: {PredictionState.WEAKENED, PredictionState.INVALIDATED},
    PredictionState.CONFIRMED: {PredictionState.COMPLETED},
    PredictionState.INVALIDATED: {PredictionState.COMPLETED},
    PredictionState.COMPLETED: set(),
}


class PredictionLifecycle:
    """Manages prediction state independently of timeline history.

    A prediction is a hypothesis about where price will go. It is NOT
    the same as a timeline — a timeline records what happened, while
    a prediction tracks how the forecast evolves relative to evidence.

    State machine:
      CREATED → OBSERVED → STRENGTHENED / WEAKENED → THREATENED
        → CONFIRMED / INVALIDATED → COMPLETED
    """

    def __init__(self, corpus: Any) -> None:
        self._corpus = corpus
        self._state: dict[str, PredictionState] = {}
        self._strength: dict[str, float] = {}

    def create(self, prediction_id: str) -> None:
        self._state[prediction_id] = PredictionState.CREATED
        self._strength[prediction_id] = 0.5
        logger.info("[PREDICTION] Created", prediction_id=prediction_id)

    def get_state(self, prediction_id: str) -> Optional[PredictionState]:
        return self._state.get(prediction_id)

    def transition(self, prediction_id: str, to_state: PredictionState) -> bool:
        current = self._state.get(prediction_id)
        if current is None:
            logger.warning("[PREDICTION] Unknown prediction", prediction_id=prediction_id)
            return False
        if to_state not in _VALID_TRANSITIONS.get(current, set()):
            logger.warning("[PREDICTION] Invalid transition",
                           prediction_id=prediction_id, from_state=current.value, to_state=to_state.value)
            return False
        self._state[prediction_id] = to_state
        logger.info("[PREDICTION] Transitioned", prediction_id=prediction_id,
                     from_state=current.value, to_state=to_state.value)
        return True

    def process_observation(self, prediction_id: str, observation: Observation) -> None:
        current = self._state.get(prediction_id)
        if current is None:
            return

        if current == PredictionState.CREATED:
            self.transition(prediction_id, PredictionState.OBSERVED)
            return

        if current in (PredictionState.OBSERVED, PredictionState.STRENGTHENED, PredictionState.WEAKENED):
            if observation.category == ObservationCategory.POSITION:
                pos_data = observation.data or {}
                pnl = pos_data.get("unrealized_pnl", 0)
                if isinstance(pnl, (int, float)):
                    self._update_strength(prediction_id, pnl)
                    if pnl > 0:
                        self.transition(prediction_id, PredictionState.STRENGTHENED)
                    elif pnl < 0:
                        self.transition(prediction_id, PredictionState.WEAKENED)

            if observation.category == ObservationCategory.RISK:
                risk_data = observation.data or {}
                if risk_data.get("stop_loss_hit") or risk_data.get("drawdown_exceeded"):
                    self.transition(prediction_id, PredictionState.THREATENED)

    def finalize(self, prediction_id: str, was_correct: bool) -> None:
        current = self._state.get(prediction_id)
        if current is None:
            return
        mid = PredictionState.CONFIRMED if was_correct else PredictionState.INVALIDATED
        if self.transition(prediction_id, mid):
            self.transition(prediction_id, PredictionState.COMPLETED)

    def _update_strength(self, prediction_id: str, pnl: float) -> float:
        current = self._strength.get(prediction_id, 0.5)
        delta = pnl * 0.01
        self._strength[prediction_id] = max(0.0, min(1.0, current + delta))
        return self._strength[prediction_id]
