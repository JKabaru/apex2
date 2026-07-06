from __future__ import annotations

from datetime import datetime

import structlog

from src.api.binance_client import BinanceClient
from src.core.events import EventBus
from src.core.models import Position, PositionState, SystemEvent
from src.db.portfolio_store import PortfolioStore
from src.models.reasoning import PortfolioSnapshot

logger = structlog.get_logger("portfolio_manager")

_VALID_TRANSITIONS: dict[PositionState, set[PositionState]] = {
    PositionState.DISCOVERED: {PositionState.VALIDATED},
    PositionState.VALIDATED: {PositionState.APPROVED, PositionState.DISCOVERED},
    PositionState.APPROVED: {PositionState.EXECUTING},
    PositionState.EXECUTING: {PositionState.OPEN},
    PositionState.OPEN: {PositionState.UNDER_REVIEW, PositionState.CLOSING},
    PositionState.UNDER_REVIEW: {PositionState.OPEN, PositionState.CLOSING, PositionState.CLOSED},
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
        live_count = sum(1 for p in self._positions.values() if p.execution_mode == "LIVE")
        shadow_count = sum(1 for p in self._positions.values() if p.execution_mode == "SHADOW")
        logger.info(
            "Crash recovery loaded positions",
            total=len(self._positions),
            live=live_count,
            shadow=shadow_count,
        )

    async def purge_stale_positions(self) -> None:
        count = len(self._positions)
        if count == 0:
            return
        logger.warning(
            "Purging stale positions from previous session",
            count=count,
        )
        for pos in self._positions.values():
            pos.lifecycle_state = PositionState.CLOSED
            pos.exit_timestamp = datetime.utcnow()
            self._store.save_position(pos)
            self._store.append_audit_log(SystemEvent(
                event_type="POSITION_PURGED_STARTUP",
                service_name="PortfolioManager",
                payload={
                    "position_id": pos.position_id,
                    "symbol": pos.symbol,
                    "reason": "STARTUP_PURGE",
                },
            ))
        self._positions.clear()
        logger.warning("Stale positions purged — exchange will be re-adopted fresh", purged=count)

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
                "execution_mode": position.execution_mode,
                "origin": position.origin,
                "quantity": position.quantity,
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
                "execution_mode": position.execution_mode,
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

    def get_live_open_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.execution_mode == "LIVE"
            and p.lifecycle_state in (PositionState.OPEN, PositionState.UNDER_REVIEW, PositionState.UNMANAGED_ADOPTED)
        ]

    def get_live_positions(self) -> list[Position]:
        return self.get_live_open_positions()

    def get_shadow_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.execution_mode == "SHADOW"
            and p.lifecycle_state in (PositionState.OPEN, PositionState.UNDER_REVIEW, PositionState.UNMANAGED_ADOPTED)
        ]

    def get_terminal_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.lifecycle_state in (PositionState.CLOSED, PositionState.ARCHIVED)
        ]

    def get_position_by_id(self, position_id: str) -> Position | None:
        return self._positions.get(position_id)

    def get_live_exposure(self, symbol: str) -> float:
        total = 0.0
        for pos in self._positions.values():
            if (
                pos.symbol == symbol
                and pos.execution_mode == "LIVE"
                and pos.lifecycle_state in (
                    PositionState.OPEN, PositionState.UNDER_REVIEW,
                    PositionState.EXECUTING, PositionState.UNMANAGED_ADOPTED
                )
            ):
                total += pos.quantity * pos.avg_fill_price
        return total

    def get_total_live_exposure(self) -> float:
        total = 0.0
        for pos in self._positions.values():
            if (
                pos.execution_mode == "LIVE"
                and pos.lifecycle_state in (
                    PositionState.OPEN, PositionState.UNDER_REVIEW,
                    PositionState.EXECUTING, PositionState.UNMANAGED_ADOPTED
                )
            ):
                total += pos.quantity * pos.avg_fill_price
        return total

    async def build_snapshot(
        self,
        max_positions: int = 3,
        min_llm_confidence: float = 0.3,
        max_live_exposure_usdt: float = 10000.0,
    ) -> PortfolioSnapshot:
        live_positions = self.get_live_open_positions()
        total_exposure = self.get_total_live_exposure()
        return PortfolioSnapshot(
            live_position_count=len(live_positions),
            live_exposure_usdt=total_exposure,
            total_live_exposure_usdt=total_exposure,
            available_margin=max(0.0, max_live_exposure_usdt - total_exposure),
            max_positions=max_positions,
            min_llm_confidence=min_llm_confidence,
            max_live_exposure_usdt=max_live_exposure_usdt,
        )

    def reload_from_database(self) -> None:
        self._positions.clear()
        self._load_existing_positions()
        live_count = sum(1 for p in self._positions.values() if p.execution_mode == "LIVE")
        shadow_count = sum(1 for p in self._positions.values() if p.execution_mode == "SHADOW")
        logger.info(
            "PortfolioManager state reloaded from database",
            total=len(self._positions),
            live=live_count,
            shadow=shadow_count,
        )

    async def reconcile(self, binance_client: BinanceClient, max_positions: int = 3) -> dict:
        from src.services.reconciler import Reconciler
        result = await Reconciler.reconcile(self, binance_client, max_positions=max_positions)

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
