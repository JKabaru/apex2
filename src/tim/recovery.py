from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Optional

import structlog

from src.core.models import Position
from src.db.tim_store import TimStore
from src.db.write_coordinator import DatabaseWriteCoordinator
from src.models.tim.enums import (
    JournalEventType,
    OriginQuality,
    TIMMode,
    ThesisStatus,
    TradeMemoryRecoveryState,
)
from src.models.tim.trade_memory import (
    TradeJournalEntry,
    TradeMemoryRecoveryRecord,
    TradeOrigin,
    WorkingMemory,
)

logger = structlog.get_logger("tim_recovery")


class TimMemoryRecoveryService:
    def __init__(
        self,
        tim_store: Optional[TimStore],
        coordinator: Optional[DatabaseWriteCoordinator],
        tim_mode: TIMMode,
        bootstrap_enabled: bool,
    ) -> None:
        self._tim_store = tim_store
        self._coordinator = coordinator
        try:
            self._tim_mode = tim_mode if isinstance(tim_mode, TIMMode) else TIMMode(tim_mode)
        except (ValueError, KeyError, TypeError):
            self._tim_mode = TIMMode.OFF
        self._bootstrap_enabled = bootstrap_enabled

    @property
    def is_active(self) -> bool:
        return self._tim_store is not None and self._tim_mode != TIMMode.OFF

    async def recover_open_positions(
        self, positions: list[Position],
    ) -> dict[str, int]:
        if not self.is_active:
            logger.info("TIM_RECOVERY_SKIPPED", reason="tim_disabled", tim_mode=self._tim_mode.value)
            return {"total": len(positions), "recovered": 0, "rebuilt": 0, "bootstrapped": 0, "skipped": len(positions)}

        total = len(positions)
        recovered = 0
        rebuilt = 0
        bootstrapped = 0
        skipped = 0

        for pos in positions:
            try:
                if pos.trade_memory_id is not None:
                    ok, did_rebuild = self._recover_single_position(pos)
                    if ok:
                        recovered += 1
                        if did_rebuild:
                            rebuilt += 1
                    else:
                        skipped += 1
                elif self._reassociate_position(pos):
                    ok, did_rebuild = self._recover_single_position(pos)
                    if ok:
                        recovered += 1
                        if did_rebuild:
                            rebuilt += 1
                    else:
                        skipped += 1
                elif self._bootstrap_enabled:
                    ok = self._bootstrap_position(pos)
                    if ok:
                        bootstrapped += 1
                    else:
                        skipped += 1
                else:
                    logger.info(
                        "TIM_RECOVERY_SKIPPED",
                        position_id=pos.position_id,
                        reason="no_trade_memory_id",
                        bootstrap_enabled=False,
                    )
                    skipped += 1
            except Exception:
                logger.error(
                    "TIM_RECOVERY_FAILED",
                    position_id=pos.position_id,
                    exc_info=True,
                )
                skipped += 1

        logger.info(
            "TIM_RECOVERY_COMPLETED",
            total=total,
            recovered=recovered,
            rebuilt=rebuilt,
            bootstrapped=bootstrapped,
            skipped=skipped,
        )
        return {
            "total": total,
            "recovered": recovered,
            "rebuilt": rebuilt,
            "bootstrapped": bootstrapped,
            "skipped": skipped,
        }

    def _reassociate_position(self, pos: Position) -> bool:
        existing = self._tim_store.get_origin_by_position(pos.position_id)
        if existing is None:
            return False
        if pos.trade_memory_id is not None:
            return True
        pos.trade_memory_id = existing.memory_id
        pos.origin_episode_id = existing.origin_episode_id
        logger.info(
            "TIM_REASSOCIATED",
            position_id=pos.position_id,
            origin_id=existing.memory_id,
        )
        journal_entry = TradeJournalEntry(
            position_id=pos.position_id,
            memory_id=existing.memory_id,
            version=1,
            event_type=JournalEventType.REASSOCIATED,
            event_data=json.dumps({
                "reason": "link_repaired",
                "origin_quality": existing.origin_quality.value,
            }),
        )
        self._tim_store.append_journal_entry(journal_entry)
        return True

    def _recover_single_position(self, pos: Position) -> tuple[bool, bool]:
        origin = self._tim_store.get_origin_by_position(pos.position_id)
        wm = self._tim_store.get_working_memory_by_position(pos.position_id)

        if origin is None or wm is None:
            logger.warning(
                "TIM_RECOVERY_INCOMPLETE",
                position_id=pos.position_id,
                origin_found=origin is not None,
                working_memory_found=wm is not None,
            )
            return False, False

        if origin.origin_episode_id:
            pos.origin_episode_id = origin.origin_episode_id

        version_before = wm.version
        valid = self._validate_checksum(wm)

        if valid:
            logger.info(
                "TIM_RECOVERY_VALID",
                position_id=pos.position_id,
                version=wm.version,
            )
            return True, False

        logger.warning(
            "TIM_RECOVERY_CHECKSUM_MISMATCH",
            position_id=pos.position_id,
            version=wm.version,
        )

        journal = self._tim_store.get_journal_entries_after_version(
            pos.position_id, 0,
        )
        rebuilt = self._rebuild_working_memory(origin, journal)
        rebuilt.metadata["crash_recovery"] = True

        self._tim_store.upsert_working_memory(rebuilt)

        journal_entry = TradeJournalEntry(
            position_id=pos.position_id,
            memory_id=rebuilt.memory_id,
            version=rebuilt.version,
            event_type=JournalEventType.WORKING_MEMORY_REBUILT,
            event_data=json.dumps({
                "reason": "checksum_mismatch",
                "old_version": version_before,
                "new_version": rebuilt.version,
            }),
        )
        self._tim_store.append_journal_entry(journal_entry)

        logger.info(
            "WORKING_MEMORY_REBUILT",
            position_id=pos.position_id,
            memory_id=rebuilt.memory_id,
            old_version=version_before,
            new_version=rebuilt.version,
            reason="checksum_mismatch",
        )
        return True, True

    def _rebuild_working_memory(
        self, origin: TradeOrigin, journal: list[TradeJournalEntry],
    ) -> WorkingMemory:
        wm = WorkingMemory(
            memory_id=origin.memory_id,
            position_id=origin.position_id,
            version=1,
            thesis_status=ThesisStatus.INTACT,
        )

        self._apply_origin_to_working_memory(wm, origin)

        for entry in journal:
            self._apply_journal_entry(wm, entry)

        wm.version = max(
            (e.version for e in journal),
            default=1,
        )
        if journal:
            wm.version += 1

        wm.checksum = self._compute_checksum(wm)
        return wm

    def _apply_origin_to_working_memory(
        self, wm: WorkingMemory, origin: TradeOrigin,
    ) -> None:
        wm.thesis_status = ThesisStatus.INTACT
        wm.review_count = 0
        wm.failed_review_count = 0

    def _apply_journal_entry(
        self, wm: WorkingMemory, entry: TradeJournalEntry,
    ) -> None:
        try:
            data = json.loads(entry.event_data) if entry.event_data != "{}" else {}
        except (json.JSONDecodeError, TypeError):
            data = {}

        if entry.event_type == JournalEventType.THESIS_UPDATED:
            status_str = data.get("new_thesis_status", "")
            try:
                wm.thesis_status = ThesisStatus(status_str)
            except ValueError:
                pass
        elif entry.event_type == JournalEventType.REVIEW_COMPLETED:
            wm.review_count += 1
        elif entry.event_type == JournalEventType.REVIEW_REQUESTED:
            pass
        elif entry.event_type == JournalEventType.PROTECTION_MODE_CHANGED:
            mode_str = data.get("new_mode", "")
            try:
                from src.models.tim.enums import ProtectionMode
                wm.protection_mode = ProtectionMode(mode_str)
            except ValueError:
                pass

    def _bootstrap_position(self, pos: Position) -> bool:
        existing_origin = self._tim_store.get_origin_by_position(pos.position_id)
        if existing_origin is not None:
            pos.trade_memory_id = existing_origin.memory_id
            pos.origin_episode_id = existing_origin.origin_episode_id
            logger.info(
                "TIM_REASSOCIATED",
                position_id=pos.position_id,
                origin_id=existing_origin.memory_id,
            )
            return True

        memory_id = str(uuid.uuid4())
        origin_episode_id = str(uuid.uuid4())

        origin = TradeOrigin(
            memory_id=memory_id,
            position_id=pos.position_id,
            origin_episode_id=origin_episode_id,
            origin_quality=OriginQuality.LOW,
            entry_thesis=pos.entry_thesis or "",
            entry_price=pos.avg_fill_price,
            entry_timestamp=pos.entry_timestamp,
            symbol=pos.symbol,
            side=pos.side,
            anchor_symbol=pos.anchor_symbol or "",
            timeframe=pos.timeframe or "",
            metadata={
                "source": "BOOTSTRAP",
                "strategy_name": pos.strategy_version or "",
                "bootstrap_reason": "pre_tim_position",
                "initial_conviction": 0.0,
            },
        )
        if pos.trade_context is not None:
            origin.metadata["expected_catalyst"] = pos.trade_context.expected_catalyst or ""
            origin.metadata["invalidation_conditions"] = pos.trade_context.expected_invalidation or ""
            origin.metadata["expected_horizon_hours"] = pos.trade_context.expected_holding_horizon_hours
            origin.metadata["scanner_name"] = pos.trade_context.scanner_name or ""
            origin.metadata["strategy_name"] = pos.trade_context.strategy_name or origin.metadata["strategy_name"]
            origin.metadata["core_hypothesis"] = pos.trade_context.thesis or ""
            origin.metadata["direction"] = pos.trade_context.direction or ""
            origin.metadata["anchor_info"] = {
                "anchor_symbol": pos.trade_context.anchor_symbol,
                "target_symbol": pos.trade_context.target_symbol,
                "relationship": pos.trade_context.relationship,
            }

        if pos.initial_evidence is not None:
            origin.entry_atr = pos.initial_evidence.atr

        self._tim_store.insert_origin(origin)

        wm = WorkingMemory(
            memory_id=memory_id,
            position_id=pos.position_id,
            version=1,
            thesis_status=ThesisStatus.INTACT,
        )
        wm.metadata["initial_conviction"] = 0.0
        wm.metadata["current_conviction"] = 0.0
        wm.metadata["conviction_source"] = "SYNTHETIC"
        wm.metadata["reference_timestamp"] = pos.entry_timestamp.isoformat() if hasattr(pos.entry_timestamp, "isoformat") else str(pos.entry_timestamp)
        wm.checksum = self._compute_checksum(wm)
        self._tim_store.upsert_working_memory(wm)

        j1 = TradeJournalEntry(
            position_id=pos.position_id,
            memory_id=memory_id,
            version=1,
            event_type=JournalEventType.ORIGIN_SYNTHETIC_RECONSTRUCTED,
            event_data=json.dumps({
                "source": "BOOTSTRAP",
                "origin_quality": "LOW",
            }),
        )
        self._tim_store.append_journal_entry(j1)
        j2 = TradeJournalEntry(
            position_id=pos.position_id,
            memory_id=memory_id,
            version=2,
            event_type=JournalEventType.WORKING_MEMORY_INITIALIZED,
            event_data=json.dumps({
                "source": "BOOTSTRAP",
                "conviction": 0.0,
                "conviction_source": "SYNTHETIC",
            }),
        )
        self._tim_store.append_journal_entry(j2)

        pos.trade_memory_id = memory_id
        pos.origin_episode_id = origin_episode_id

        logger.info(
            "ORIGIN_SYNTHETIC_RECONSTRUCTED",
            position_id=pos.position_id,
            origin_id=memory_id,
            source="BOOTSTRAP",
        )
        logger.info(
            "WORKING_MEMORY_INITIALIZED",
            memory_id=memory_id,
            origin_id=memory_id,
            position_id=pos.position_id,
            version=1,
        )
        return True

    def _compute_checksum(self, wm: WorkingMemory) -> str:
        data = {
            "memory_id": wm.memory_id,
            "position_id": wm.position_id,
            "version": wm.version,
            "thesis_status": wm.thesis_status.value if wm.thesis_status else "",
            "protection_mode": wm.protection_mode.value if wm.protection_mode else "",
            "review_count": wm.review_count,
            "failed_review_count": wm.failed_review_count,
            "metadata": wm.metadata,
        }
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _validate_checksum(self, wm: WorkingMemory) -> bool:
        if not wm.checksum:
            return wm.version <= 1
        expected = self._compute_checksum(wm)
        return expected == wm.checksum
