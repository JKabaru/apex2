from __future__ import annotations

from datetime import datetime

import structlog

from src.api.binance_client import BinanceClient
from src.core.events import EventBus
from src.core.models import Position, PositionState, SystemEvent
from src.db.portfolio_store import PortfolioStore

logger = structlog.get_logger("portfolio_manager")

_VALID_TRANSITIONS: dict[PositionState, set[PositionState]] = {
    PositionState.DISCOVERED: {PositionState.VALIDATED},
    PositionState.VALIDATED: {PositionState.APPROVED, PositionState.DISCOVERED},
    PositionState.APPROVED: {PositionState.EXECUTING},
    PositionState.EXECUTING: {PositionState.OPEN},
    PositionState.OPEN: {PositionState.UNDER_REVIEW, PositionState.CLOSING},
    PositionState.UNDER_REVIEW: {PositionState.OPEN, PositionState.CLOSING},
    PositionState.UNMANAGED_ADOPTED: {PositionState.CLOSING, PositionState.CLOSED, PositionState.OPEN},
    PositionState.CLOSING: {PositionState.CLOSED},
    PositionState.CLOSED: {PositionState.ARCHIVED},
    PositionState.ARCHIVED: set(),
}


class PortfolioManager:
    def __init__(self, store: PortfolioStore, event_bus: EventBus):
        self._store = store
        self._event_bus = event_bus
        self._positions: dict[str, Position] = {}
        self._load_existing_positions()

    def _load_existing_positions(self) -> None:
        all_positions = self._store.get_all_positions()
        active_states = {PositionState.OPEN, PositionState.UNDER_REVIEW, PositionState.UNMANAGED_ADOPTED}
        for pos in all_positions:
            if pos.lifecycle_state in active_states:
                self._positions[pos.position_id] = pos
        logger.info("Crash recovery loaded positions", count=len(self._positions))

    async def add_position(self, position: Position) -> None:
        self._positions[position.position_id] = position
        self._store.save_position(position)
        event = SystemEvent(
            event_type="POSITION_OPENED",
            service_name="PortfolioManager",
            payload={
                "position_id": position.position_id,
                "symbol": position.symbol,
                "side": position.side,
                "lifecycle_state": position.lifecycle_state.value,
            },
        )
        self._store.append_audit_log(event)
        await self._event_bus.publish(event)

    async def update_position_state(
        self, position_id: str, new_state: PositionState, **kwargs
    ) -> None:
        position = self._positions.get(position_id)
        if position is None:
            raise ValueError(f"Position {position_id} not found")

        current = position.lifecycle_state
        allowed = _VALID_TRANSITIONS.get(current, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition: {current.value} -> {new_state.value}. "
                f"Allowed from {current.value}: {[s.value for s in allowed]}"
            )

        position.lifecycle_state = new_state
        for key, value in kwargs.items():
            if hasattr(position, key):
                setattr(position, key, value)

        if new_state in (PositionState.CLOSED, PositionState.ARCHIVED):
            position.exit_timestamp = datetime.utcnow()

        self._store.save_position(position)
        event = SystemEvent(
            event_type="POSITION_UPDATED",
            service_name="PortfolioManager",
            payload={
                "position_id": position_id,
                "symbol": position.symbol,
                "previous_state": current.value,
                "new_state": new_state.value,
                **kwargs,
            },
        )
        self._store.append_audit_log(event)
        await self._event_bus.publish(event)

    def get_open_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.lifecycle_state in (PositionState.OPEN, PositionState.UNDER_REVIEW, PositionState.UNMANAGED_ADOPTED)
        ]

    async def reconcile(self, binance_client: BinanceClient) -> dict:
        from src.services.reconciler import Reconciler
        result = await Reconciler.reconcile(self, binance_client)

        for detail in result.get("details", []):
            detail_type = detail.get("type")
            if detail_type == "ADOPTED":
                self._store.append_audit_log(SystemEvent(
                    event_type="POSITION_ADOPTED",
                    service_name="PortfolioManager",
                    payload={k: v for k, v in detail.items() if k != "type"},
                ))
            elif detail_type == "ADOPTION_FAILED":
                event = SystemEvent(
                    event_type="UNMANAGED_POSITION_DETECTED",
                    service_name="PortfolioManager",
                    payload={k: v for k, v in detail.items() if k != "type"},
                )
                self._store.append_audit_log(event)
                await self._event_bus.publish(event)

        return result

    def get_exposure(self, symbol: str) -> float:
        total = 0.0
        for pos in self._positions.values():
            if pos.symbol == symbol and pos.lifecycle_state in (
                PositionState.OPEN, PositionState.UNDER_REVIEW, PositionState.EXECUTING, PositionState.UNMANAGED_ADOPTED
            ):
                total += pos.quantity * pos.avg_fill_price
        return total
