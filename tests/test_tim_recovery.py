from __future__ import annotations

import inspect
from datetime import datetime

import duckdb
import pytest

from src.core.models import Position, PositionState, ProtectionOrders
from src.db.tim_store import TimStore
from src.db.write_coordinator import DatabaseWriteCoordinator
from src.models.tim.enums import TIMMode, OriginQuality, ThesisStatus, JournalEventType
from src.models.tim.trade_memory import TradeOrigin, WorkingMemory
from src.tim.recovery import TimMemoryRecoveryService


@pytest.fixture
def tim_store():
    conn = duckdb.connect(":memory:")
    store = TimStore(connection=conn)
    store.create_schema()
    return store


@pytest.fixture
def coordinator(tim_store):
    return DatabaseWriteCoordinator(tim_store._conn)


@pytest.fixture
def recovery_service(tim_store, coordinator):
    return TimMemoryRecoveryService(
        tim_store=tim_store,
        coordinator=coordinator,
        tim_mode=TIMMode.MEMORY_ONLY,
        bootstrap_enabled=True,
    )


def make_position(position_id="pos-1", symbol="BTCUSDT", side="LONG",
                  qty=0.5, price=50000.0, stop=49000.0, tp=55000.0):
    return Position(
        position_id=position_id,
        symbol=symbol,
        side=side,
        quantity=qty,
        avg_fill_price=price,
        anchor_symbol=symbol.replace("USDT", ""),
        entry_timestamp=datetime.utcnow(),
        timeframe="5m",
        lifecycle_state=PositionState.OPEN,
        protection_orders=ProtectionOrders(stop_price=stop, tp_price=tp),
    )


class TestRecoveryConstructorSignature:
    def test_fixture_matches_real_signature(self):
        sig = inspect.signature(TimMemoryRecoveryService.__init__)
        params = list(sig.parameters.keys())
        assert "tim_store" in params
        assert "coordinator" in params
        assert "tim_mode" in params
        assert "bootstrap_enabled" in params


class TestRecoveryMode:
    @pytest.mark.asyncio
    async def test_skips_when_off(self, tim_store, coordinator):
        svc = TimMemoryRecoveryService(
            tim_store=tim_store, coordinator=coordinator,
            tim_mode=TIMMode.OFF, bootstrap_enabled=True,
        )
        result = await svc.recover_open_positions([])
        assert result["recovered"] == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_positions(self, recovery_service):
        result = await recovery_service.recover_open_positions([])
        assert result["total"] == 0
        assert result["recovered"] == 0

    def test_invalid_mode_defaults_to_off(self, tim_store, coordinator):
        svc = TimMemoryRecoveryService(
            tim_store=tim_store, coordinator=coordinator,
            tim_mode="INVALID_MODE", bootstrap_enabled=True,
        )
        assert svc._tim_mode == TIMMode.OFF


class TestRecoveryWithExistingMemory:
    @pytest.mark.asyncio
    async def test_reassociates_when_origin_exists(self, tim_store, recovery_service):
        pos = make_position("pos-recovery-1")
        tim_store.insert_origin(TradeOrigin(
            position_id="pos-recovery-1", origin_episode_id="ep-1",
            symbol="BTCUSDT", side="LONG", memory_id="mem-recovery-1",
            entry_price=50000.0, entry_atr=1000.0, origin_quality=OriginQuality.LOW,
        ))
        wm = WorkingMemory(
            memory_id="mem-recovery-1", position_id="pos-recovery-1", version=1,
        )
        wm.checksum = recovery_service._compute_checksum(wm)
        tim_store.upsert_working_memory(wm)

        result = await recovery_service.recover_open_positions([pos])
        assert result["recovered"] == 1
        assert pos.trade_memory_id == "mem-recovery-1"
        assert pos.origin_episode_id == "ep-1"

        entries = tim_store.get_journal_entries("pos-recovery-1")
        reassoc_events = [e for e in entries if e.event_type == JournalEventType.REASSOCIATED]
        assert len(reassoc_events) == 1
        assert reassoc_events[0].event_type == JournalEventType.REASSOCIATED

    @pytest.mark.asyncio
    async def test_bootstraps_when_no_origin(self, tim_store, recovery_service):
        pos = make_position("pos-recovery-2", symbol="ETHUSDT", side="SHORT",
                            price=3000.0, stop=3100.0, tp=2800.0)
        result = await recovery_service.recover_open_positions([pos])
        assert result["bootstrapped"] == 1
        assert pos.trade_memory_id is not None
        assert pos.origin_episode_id is not None

        origin = tim_store.get_origin_by_position("pos-recovery-2")
        assert origin is not None
        assert origin.origin_quality == OriginQuality.LOW

        entries = tim_store.get_journal_entries("pos-recovery-2")
        event_types = [e.event_type for e in entries]
        assert JournalEventType.ORIGIN_SYNTHETIC_RECONSTRUCTED in event_types
        assert JournalEventType.WORKING_MEMORY_INITIALIZED in event_types

    @pytest.mark.asyncio
    async def test_skips_when_bootstrap_disabled_and_no_origin(self, tim_store, coordinator):
        svc = TimMemoryRecoveryService(
            tim_store=tim_store, coordinator=coordinator,
            tim_mode=TIMMode.MEMORY_ONLY, bootstrap_enabled=False,
        )
        pos = make_position("pos-recovery-3", symbol="XRPUSDT",
                            price=0.50, stop=0.45, tp=0.60)
        result = await svc.recover_open_positions([pos])
        assert result["skipped"] == 1

    @pytest.mark.asyncio
    async def test_checksum_valid_does_not_rebuild(self, tim_store, recovery_service):
        pos = make_position("pos-checksum-1")
        pos.trade_memory_id = "mem-checksum-1"
        tim_store.insert_origin(TradeOrigin(
            position_id="pos-checksum-1", origin_episode_id="ep-checksum-1",
            symbol="BTCUSDT", side="LONG", memory_id="mem-checksum-1",
            entry_price=50000.0, entry_atr=1000.0, origin_quality=OriginQuality.LOW,
        ))
        wm = WorkingMemory(
            memory_id="mem-checksum-1", position_id="pos-checksum-1", version=1,
        )
        wm.checksum = recovery_service._compute_checksum(wm)
        tim_store.upsert_working_memory(wm)

        result = await recovery_service.recover_open_positions([pos])
        assert result["recovered"] == 1
        assert result["rebuilt"] == 0

    @pytest.mark.asyncio
    async def test_checksum_mismatch_rebuilds(self, tim_store, recovery_service):
        pos = make_position("pos-checksum-2")
        pos.trade_memory_id = "mem-checksum-2"
        origin = TradeOrigin(
            position_id="pos-checksum-2", origin_episode_id="ep-checksum-2",
            symbol="BTCUSDT", side="LONG", memory_id="mem-checksum-2",
            entry_price=50000.0, entry_atr=1000.0, origin_quality=OriginQuality.LOW,
        )
        tim_store.insert_origin(origin)
        tim_store.upsert_working_memory(WorkingMemory(
            memory_id="mem-checksum-2", position_id="pos-checksum-2",
            version=1, checksum="invalid-checksum",
        ))

        result = await recovery_service.recover_open_positions([pos])
        assert result["rebuilt"] == 1

        entries = tim_store.get_journal_entries("pos-checksum-2")
        rebuild_events = [e for e in entries if e.event_type == JournalEventType.WORKING_MEMORY_REBUILT]
        assert len(rebuild_events) >= 1

    @pytest.mark.asyncio
    async def test_rebuild_sets_crash_flag_in_metadata(self, tim_store, recovery_service):
        pos = make_position("pos-crash-flag")
        pos.trade_memory_id = "mem-crash-flag"
        tim_store.insert_origin(TradeOrigin(
            position_id="pos-crash-flag", origin_episode_id="ep-crash",
            symbol="BTCUSDT", side="LONG", memory_id="mem-crash-flag",
            entry_price=50000.0, entry_atr=1000.0, origin_quality=OriginQuality.LOW,
        ))
        tim_store.upsert_working_memory(WorkingMemory(
            memory_id="mem-crash-flag", position_id="pos-crash-flag",
            version=1, checksum="tampered",
        ))

        await recovery_service.recover_open_positions([pos])

        wm_after = tim_store.get_working_memory_by_position("pos-crash-flag")
        assert wm_after.metadata.get("crash_recovery") is True

        entries = tim_store.get_journal_entries("pos-crash-flag")
        rebuild_events = [e for e in entries if e.event_type == JournalEventType.WORKING_MEMORY_REBUILT]
        assert len(rebuild_events) == 1

    @pytest.mark.asyncio
    async def test_clean_recovery_twice_no_duplicate_journals(self, tim_store, recovery_service):
        pos = make_position("pos-clean-twice")
        pos.trade_memory_id = "mem-clean-twice"
        tim_store.insert_origin(TradeOrigin(
            position_id="pos-clean-twice", origin_episode_id="ep-clean",
            symbol="BTCUSDT", side="LONG", memory_id="mem-clean-twice",
            entry_price=50000.0, entry_atr=1000.0, origin_quality=OriginQuality.LOW,
        ))
        wm = WorkingMemory(
            memory_id="mem-clean-twice", position_id="pos-clean-twice", version=1,
        )
        wm.checksum = recovery_service._compute_checksum(wm)
        tim_store.upsert_working_memory(wm)

        entries_before = len(tim_store.get_journal_entries("pos-clean-twice"))
        await recovery_service.recover_open_positions([pos])
        await recovery_service.recover_open_positions([pos])
        entries_after = len(tim_store.get_journal_entries("pos-clean-twice"))
        assert entries_after == entries_before

    @pytest.mark.asyncio
    async def test_recovery_with_position_having_trade_memory_id(self, tim_store, recovery_service):
        pos = make_position("pos-memid")
        pos.trade_memory_id = "mem-already-set"
        tim_store.insert_origin(TradeOrigin(
            position_id="pos-memid", origin_episode_id="ep-memid",
            symbol="BTCUSDT", side="LONG", memory_id="mem-already-set",
            entry_price=50000.0, entry_atr=1000.0, origin_quality=OriginQuality.LOW,
        ))
        wm = WorkingMemory(
            memory_id="mem-already-set", position_id="pos-memid", version=1,
        )
        wm.checksum = recovery_service._compute_checksum(wm)
        tim_store.upsert_working_memory(wm)
        result = await recovery_service.recover_open_positions([pos])
        assert result["recovered"] == 1

    @pytest.mark.asyncio
    async def test_upsert_working_memory_idempotent(self, tim_store, recovery_service):
        pos = make_position("pos-upsert-idem")
        pos.trade_memory_id = "mem-upsert-idem"
        tim_store.insert_origin(TradeOrigin(
            position_id="pos-upsert-idem", origin_episode_id="ep-upsert",
            symbol="BTCUSDT", side="LONG", memory_id="mem-upsert-idem",
            entry_price=50000.0, entry_atr=1000.0, origin_quality=OriginQuality.LOW,
        ))
        wm = WorkingMemory(
            memory_id="mem-upsert-idem", position_id="pos-upsert-idem", version=1,
        )
        wm.checksum = recovery_service._compute_checksum(wm)
        tim_store.upsert_working_memory(wm)
        tim_store.upsert_working_memory(wm)
        wm2 = tim_store.get_working_memory_by_position("pos-upsert-idem")
        assert wm2.version == 1


class TestRecoveryStoreUnavailable:
    @pytest.mark.asyncio
    async def test_startup_succeeds_when_store_unavailable(self, coordinator):
        svc = TimMemoryRecoveryService(
            tim_store=None, coordinator=coordinator,
            tim_mode=TIMMode.MEMORY_ONLY, bootstrap_enabled=True,
        )
        result = await svc.recover_open_positions([make_position("pos-nostore")])
        assert result["recovered"] == 0


class TestComputeChecksum:
    def test_checksum_is_deterministic(self, recovery_service):
        wm1 = WorkingMemory(memory_id="m1", position_id="p1", version=1)
        wm2 = WorkingMemory(memory_id="m1", position_id="p1", version=1)
        assert recovery_service._compute_checksum(wm1) == recovery_service._compute_checksum(wm2)

    def test_checksum_differs_when_data_changes(self, recovery_service):
        wm1 = WorkingMemory(memory_id="m1", position_id="p1", version=1, checksum="x")
        wm2 = WorkingMemory(memory_id="m1", position_id="p2", version=1, checksum="x")
        assert recovery_service._compute_checksum(wm1) != recovery_service._compute_checksum(wm2)
