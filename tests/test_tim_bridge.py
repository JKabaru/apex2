from __future__ import annotations

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import duckdb
import pytest

from src.db.tim_store import TimStore
from src.db.write_coordinator import DatabaseWriteCoordinator
from src.models.tim.enums import TIMMode, OriginQuality, ThesisStatus, JournalEventType
from src.models.tim.trade_memory import TradeOrigin, WorkingMemory
from src.models.tim.review import ReviewConditions
from src.tim.bridge import TimMemoryBridge


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
def bridge(coordinator, tim_store):
    return TimMemoryBridge(
        tim_store=tim_store,
        coordinator=coordinator,
        tim_mode=TIMMode.MEMORY_ONLY,
    )


@pytest.fixture
def position():
    from src.core.models import Position, PositionState, TradeContext, ProtectionOrders
    return Position(
        position_id="pos-bridge-1",
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.5,
        avg_fill_price=50000.0,
        anchor_symbol="BTC",
        entry_thesis="BTC momentum breakout",
        entry_timestamp=datetime.utcnow(),
        timeframe="5m",
        strategy_version="1.0",
        execution_model="fixed_friction_v1",
        lifecycle_state=PositionState.OPEN,
        protection_orders=ProtectionOrders(
            stop_price=49000.0,
            tp_price=55000.0,
        ),
        trade_context=TradeContext(
            anchor_symbol="BTC",
            direction="LONG",
            thesis="BTC momentum breakout",
            expected_catalyst="ETF approval",
            expected_invalidation="Regulatory rejection",
            expected_holding_horizon_hours=48.0,
            scanner_name="momentum_scanner",
            strategy_name="momentum_breakout_v1",
        ),
    )


@pytest.fixture
def position_no_context():
    from src.core.models import Position, PositionState, ProtectionOrders
    return Position(
        position_id="pos-bridge-2",
        symbol="ETHUSDT",
        side="SHORT",
        quantity=1.0,
        avg_fill_price=3000.0,
        anchor_symbol="ETH",
        entry_timestamp=datetime.utcnow(),
        timeframe="15m",
        lifecycle_state=PositionState.OPEN,
        protection_orders=ProtectionOrders(
            stop_price=3100.0,
            tp_price=2800.0,
        ),
    )


class TestBridgeConstructorSignature:
    def test_fixture_matches_real_signature(self):
        sig = inspect.signature(TimMemoryBridge.__init__)
        params = list(sig.parameters.keys())
        assert "tim_store" in params
        assert "coordinator" in params
        assert "tim_mode" in params


class TestBridgeEnabled:
    def test_disabled_when_off(self, tim_store, coordinator):
        b = TimMemoryBridge(tim_store=tim_store, coordinator=coordinator, tim_mode=TIMMode.OFF)
        assert not b.is_enabled()

    def test_enabled_when_memory_only(self, tim_store, coordinator):
        b = TimMemoryBridge(tim_store=tim_store, coordinator=coordinator, tim_mode=TIMMode.MEMORY_ONLY)
        assert b.is_enabled()

    def test_disabled_when_store_none(self, coordinator):
        b = TimMemoryBridge(tim_store=None, coordinator=coordinator, tim_mode=TIMMode.MEMORY_ONLY)
        assert not b.is_enabled()


class TestConfigLoading:
    def test_valid_mode_from_config(self, tim_store):
        from src.models.tim.config import TIMConfig
        config = TIMConfig(tim_mode="MEMORY_ONLY")
        b = TimMemoryBridge(tim_store=tim_store, coordinator=MagicMock(), tim_mode=config.tim_mode)
        assert b.is_enabled()

    def test_invalid_mode_defaults_to_off(self):
        b = TimMemoryBridge(tim_store=None, coordinator=None, tim_mode="INVALID_MODE")
        assert not b.is_enabled()


class TestOnPositionFilled:
    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, tim_store, coordinator, position):
        b = TimMemoryBridge(tim_store=tim_store, coordinator=coordinator, tim_mode=TIMMode.OFF)
        result = await b.on_position_filled(position)
        assert result is False
        origin = tim_store.get_origin_by_position(position.position_id)
        assert origin is None

    @pytest.mark.asyncio
    async def test_skips_when_already_has_memory(self, bridge, position):
        position.trade_memory_id = "existing-memory-id"
        result = await bridge.on_position_filled(position)
        assert result is False

    @pytest.mark.asyncio
    async def test_creates_origin_and_working_memory(self, bridge, position):
        result = await bridge.on_position_filled(position, state={"trend_regime": "BULLISH"})
        assert result is True
        assert position.trade_memory_id is not None
        assert position.origin_episode_id is not None

        origin = bridge._tim_store.get_origin_by_position(position.position_id)
        assert origin is not None
        assert origin.symbol == "BTCUSDT"
        assert origin.side == "LONG"
        assert origin.entry_price == 50000.0
        assert origin.entry_thesis == "BTC momentum breakout"

        wm = bridge._tim_store.get_working_memory_by_position(position.position_id)
        assert wm is not None
        assert wm.memory_id == origin.memory_id
        assert wm.thesis_status == ThesisStatus.INTACT
        assert wm.version == 1

    @pytest.mark.asyncio
    async def test_sets_trade_memory_id_on_position(self, bridge, position):
        await bridge.on_position_filled(position)
        assert position.trade_memory_id is not None
        assert position.origin_episode_id is not None

        wm = bridge._tim_store.get_working_memory_by_position(position.position_id)
        assert wm is not None
        assert position.trade_memory_id == wm.memory_id

        origin = bridge._tim_store.get_origin_by_position(position.position_id)
        assert origin is not None
        assert position.origin_episode_id == origin.origin_episode_id

    @pytest.mark.asyncio
    async def test_creates_journal_entries(self, bridge, position):
        await bridge.on_position_filled(position)
        entries = bridge._tim_store.get_journal_entries(position.position_id)
        assert len(entries) >= 2
        event_types = [e.event_type for e in entries]
        assert JournalEventType.ORIGIN_SET in event_types
        assert JournalEventType.WORKING_MEMORY_INITIALIZED in event_types

    @pytest.mark.asyncio
    async def test_origin_quality_medium_with_llm(self, bridge, position):
        position.llm_request_id = "llm-req-123"
        await bridge.on_position_filled(position)
        origin = bridge._tim_store.get_origin_by_position(position.position_id)
        assert origin.origin_quality == OriginQuality.MEDIUM

    @pytest.mark.asyncio
    async def test_origin_quality_medium_with_context(self, bridge, position):
        await bridge.on_position_filled(position)
        origin = bridge._tim_store.get_origin_by_position(position.position_id)
        assert origin.origin_quality == OriginQuality.MEDIUM

    @pytest.mark.asyncio
    async def test_origin_quality_low_execution_only(self, bridge, position_no_context):
        await bridge.on_position_filled(position_no_context)
        origin = bridge._tim_store.get_origin_by_position(position_no_context.position_id)
        assert origin.origin_quality == OriginQuality.LOW

    @pytest.mark.asyncio
    async def test_conviction_with_llm_and_context(self, bridge, position):
        position.llm_request_id = "llm-req-123"
        await bridge.on_position_filled(position)
        wm = bridge._tim_store.get_working_memory_by_position(position.position_id)
        assert wm.metadata["initial_conviction"] == 0.3
        assert wm.metadata["current_conviction"] == 0.3
        assert wm.metadata["conviction_source"] == "THESIS_ONLY"

    @pytest.mark.asyncio
    async def test_conviction_llm_without_context(self, bridge):
        from src.core.models import Position, PositionState, ProtectionOrders
        pos = Position(
            position_id="pos-llm-only",
            symbol="SOLUSDT", side="LONG",
            quantity=10, avg_fill_price=150.0,
            anchor_symbol="SOL", entry_timestamp=datetime.utcnow(),
            lifecycle_state=PositionState.OPEN,
            protection_orders=ProtectionOrders(stop_price=140.0, tp_price=180.0),
            llm_request_id="llm-req-456",
        )
        await bridge.on_position_filled(pos)
        wm = bridge._tim_store.get_working_memory_by_position("pos-llm-only")
        assert wm.metadata["initial_conviction"] == 0.0
        assert wm.metadata["conviction_source"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_conviction_with_context_only(self, bridge, position):
        await bridge.on_position_filled(position)
        wm = bridge._tim_store.get_working_memory_by_position(position.position_id)
        assert wm.metadata["initial_conviction"] == 0.3
        assert wm.metadata["conviction_source"] == "THESIS_ONLY"

    @pytest.mark.asyncio
    async def test_conviction_zero_for_low_quality(self, bridge, position_no_context):
        await bridge.on_position_filled(position_no_context)
        wm = bridge._tim_store.get_working_memory_by_position(position_no_context.position_id)
        assert wm.metadata.get("initial_conviction") == 0.0
        assert wm.metadata.get("conviction_source") == "SYNTHETIC"

    @pytest.mark.asyncio
    async def test_review_conditions_populated(self, bridge, position):
        await bridge.on_position_filled(position, state={"trend_regime": "BULLISH"})
        wm = bridge._tim_store.get_working_memory_by_position(position.position_id)
        conditions = ReviewConditions.model_validate_json(wm.next_review_conditions)
        assert conditions.reference_price == 50000.0
        assert conditions.reference_timestamp is not None

    @pytest.mark.asyncio
    async def test_origin_metadata_from_trade_context(self, bridge, position):
        await bridge.on_position_filled(position)
        origin = bridge._tim_store.get_origin_by_position(position.position_id)
        assert origin.metadata.get("expected_catalyst") == "ETF approval"
        assert origin.metadata.get("invalidation_conditions") == "Regulatory rejection"
        assert origin.metadata.get("expected_horizon_hours") == 48.0
        assert origin.metadata.get("scanner_name") == "momentum_scanner"
        assert origin.metadata.get("strategy_name") == "momentum_breakout_v1"

    @pytest.mark.asyncio
    async def test_origin_metadata_safely_missing(self, bridge, position_no_context):
        await bridge.on_position_filled(position_no_context)
        origin = bridge._tim_store.get_origin_by_position(position_no_context.position_id)
        assert origin.metadata.get("expected_catalyst") == ""
        assert origin.metadata.get("invalidation_conditions") == ""

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(self, bridge, position):
        bridge._tim_store = None
        result = await bridge.on_position_filled(position)
        assert result is False

    @pytest.mark.asyncio
    async def test_reference_timestamp_in_conditions(self, bridge, position):
        await bridge.on_position_filled(position)
        wm = bridge._tim_store.get_working_memory_by_position(position.position_id)
        conditions = ReviewConditions.model_validate_json(wm.next_review_conditions)
        assert conditions.reference_timestamp is not None
        assert conditions.reference_timestamp == position.entry_timestamp

    @pytest.mark.asyncio
    async def test_on_position_filled_twice_no_duplicate(self, bridge, position):
        result1 = await bridge.on_position_filled(position)
        assert result1 is True
        memory_id_1 = position.trade_memory_id
        origin_ep_1 = position.origin_episode_id

        result2 = await bridge.on_position_filled(position)
        assert result2 is False
        assert position.trade_memory_id == memory_id_1
        assert position.origin_episode_id == origin_ep_1

        entries = bridge._tim_store.get_journal_entries(position.position_id)
        assert len(entries) == 2
