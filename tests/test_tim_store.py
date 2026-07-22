from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pytest

from src.models.tim.config import TIMConfig
from src.models.tim.enums import (
    IntentStatus,
    IntentType,
    JournalEventType,
    OriginQuality,
    ProtectionMode,
    ReviewStatus,
    ReviewTriggerType,
    ThesisStatus,
    TIMMode,
    TriggerType,
)
from src.models.tim.intent import IntentExecutionRecord
from src.models.tim.review import ReviewConditions, ReviewSchedule, ReviewSession, ReviewTrigger
from src.models.tim.trade_memory import (
    JournalCompressionSummary,
    TradeJournalEntry,
    TradeOrigin,
    WorkingMemory,
)
from src.db.tim_store import TimStore


@pytest.fixture
def conn():
    db = duckdb.connect(":memory:")
    yield db
    db.close()


@pytest.fixture
def store(conn):
    s = TimStore(conn)
    s.create_schema()
    return s


@pytest.fixture
def sample_origin():
    return TradeOrigin(
        position_id="pos-1",
        origin_episode_id="ep-1",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        anchor_symbol="BTC",
        timeframe="5m",
    )


@pytest.fixture
def sample_working():
    return WorkingMemory(
        memory_id="mem-1",
        position_id="pos-1",
        version=1,
        checksum="abc123",
        thesis_status=ThesisStatus.INTACT,
        protection_mode=ProtectionMode.MECHANICAL_ONLY,
    )


@pytest.fixture
def sample_journal_entry():
    return TradeJournalEntry(
        position_id="pos-1",
        memory_id="mem-1",
        version=1,
        event_type=JournalEventType.SYSTEM_CREATED,
        event_data=json.dumps({"key": "value"}),
    )


class TestSchema:
    def test_create_schema(self, conn):
        t = TimStore(conn)
        assert t.create_schema() is True
        assert t.is_available() is True

    def test_create_schema_idempotent(self, store):
        assert store.create_schema() is True
        assert store.is_available() is True

    def test_is_available_false(self, conn):
        t = TimStore(conn)
        assert t.is_available() is False

    def test_assert_available_raises(self, conn):
        t = TimStore(conn)
        with pytest.raises(RuntimeError, match="TIM schema"):
            t.assert_available()

    def test_assert_available_ok(self, store):
        store.assert_available()


class TestTradeOrigin:
    def test_insert_and_get(self, store, sample_origin):
        assert store.insert_origin(sample_origin) is True
        loaded = store.get_origin_by_position("pos-1")
        assert loaded is not None
        assert loaded.position_id == "pos-1"
        assert loaded.origin_episode_id == "ep-1"
        assert loaded.symbol == "BTCUSDT"

    def test_insert_duplicate(self, store, sample_origin):
        assert store.insert_origin(sample_origin) is True
        assert store.insert_origin(sample_origin) is False

    def test_get_nonexistent(self, store):
        assert store.get_origin_by_position("nonexistent") is None

    def test_get_by_episode(self, store, sample_origin):
        store.insert_origin(sample_origin)
        loaded = store.get_origin_by_episode("ep-1")
        assert loaded is not None
        assert loaded.position_id == "pos-1"


class TestWorkingMemory:
    def test_upsert_and_get(self, store, sample_working):
        assert store.upsert_working_memory(sample_working) is True
        loaded = store.get_working_memory("mem-1")
        assert loaded is not None
        assert loaded.version == 1
        assert loaded.thesis_status == ThesisStatus.INTACT

    def test_upsert_update(self, store, sample_working):
        assert store.upsert_working_memory(sample_working) is True
        sample_working.thesis_status = ThesisStatus.STRENGTHENED
        sample_working.version = 2
        assert store.upsert_working_memory(sample_working) is True
        loaded = store.get_working_memory("mem-1")
        assert loaded.thesis_status == ThesisStatus.STRENGTHENED
        assert loaded.version == 2

    def test_get_by_position(self, store, sample_working):
        store.upsert_working_memory(sample_working)
        loaded = store.get_working_memory_by_position("pos-1")
        assert loaded is not None
        assert loaded.memory_id == "mem-1"

    def test_get_version(self, store, sample_working):
        store.upsert_working_memory(sample_working)
        assert store.get_working_memory_version("mem-1") == 1

    def test_get_checksum(self, store, sample_working):
        store.upsert_working_memory(sample_working)
        assert store.get_working_memory_checksum("mem-1") == "abc123"


class TestJournal:
    def test_append_and_get(self, store, sample_origin, sample_working, sample_journal_entry):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        assert store.append_journal_entry(sample_journal_entry) is True
        entries = store.get_journal_entries("pos-1")
        assert len(entries) == 1
        assert entries[0].version == 1

    def test_version_monotonic(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        e1 = TradeJournalEntry(
            position_id="pos-1", memory_id="mem-1", version=1, event_type=JournalEventType.SYSTEM_CREATED
        )
        e2 = TradeJournalEntry(
            position_id="pos-1", memory_id="mem-1", version=2, event_type=JournalEventType.REVIEW_REQUESTED
        )
        e3 = TradeJournalEntry(
            position_id="pos-1", memory_id="mem-1", version=3, event_type=JournalEventType.REVIEW_COMPLETED
        )
        assert store.append_journal_entry(e1) is True
        assert store.append_journal_entry(e2) is True
        assert store.append_journal_entry(e3) is True
        assert len(store.get_journal_entries("pos-1")) == 3

    def test_version_duplicate_rejected(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        e1 = TradeJournalEntry(
            position_id="pos-1", memory_id="mem-1", version=1, event_type=JournalEventType.SYSTEM_CREATED
        )
        e1dup = TradeJournalEntry(
            position_id="pos-1", memory_id="mem-1", version=1, event_type=JournalEventType.REVIEW_REQUESTED
        )
        assert store.append_journal_entry(e1) is True
        assert store.append_journal_entry(e1dup) is False

    def test_get_after_version(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        for v in range(1, 5):
            entry = TradeJournalEntry(
                position_id="pos-1",
                memory_id="mem-1",
                version=v,
                event_type=JournalEventType.SYSTEM_CREATED,
            )
            store.append_journal_entry(entry)
        after = store.get_journal_entries_after_version("pos-1", 2)
        assert len(after) == 2
        assert [e.version for e in after] == [3, 4]

    def test_entry_count(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        assert store.get_journal_entry_count("pos-1") == 0
        entry = TradeJournalEntry(
            position_id="pos-1", memory_id="mem-1", version=1, event_type=JournalEventType.SYSTEM_CREATED
        )
        store.append_journal_entry(entry)
        assert store.get_journal_entry_count("pos-1") == 1

    def test_latest_version(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        for v in range(1, 4):
            entry = TradeJournalEntry(
                position_id="pos-1",
                memory_id="mem-1",
                version=v,
                event_type=JournalEventType.SYSTEM_CREATED,
            )
            store.append_journal_entry(entry)
        assert store.get_latest_journal_version("pos-1") == 3


class TestCompression:
    def test_insert_and_latest(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        summary = JournalCompressionSummary(
            position_id="pos-1",
            memory_id="mem-1",
            version_start=1,
            version_end=10,
            original_entry_count=10,
        )
        assert store.insert_compression_summary(summary) is True
        loaded = store.get_latest_compression("pos-1")
        assert loaded is not None
        assert loaded.version_end == 10
        assert loaded.original_entry_count == 10

    def test_get_compression_for_version(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        summary = JournalCompressionSummary(
            position_id="pos-1",
            memory_id="mem-1",
            version_start=1,
            version_end=10,
            original_entry_count=10,
        )
        store.insert_compression_summary(summary)
        loaded = store.get_compression_for_version("pos-1", 5)
        assert loaded is not None
        assert loaded.version_start == 1

    def test_get_compression_not_found(self, store):
        assert store.get_latest_compression("nonexistent") is None


class TestReviewSession:
    def test_insert_and_get(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        session = ReviewSession(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
        )
        assert store.insert_review_session(session) is True
        loaded = store.get_review_session(session.session_id)
        assert loaded is not None
        assert loaded.status == ReviewStatus.PENDING

    def test_update_session(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        session = ReviewSession(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
        )
        store.insert_review_session(session)
        session.status = ReviewStatus.COMPLETED
        session.thesis_status_after = ThesisStatus.STRENGTHENED
        assert store.update_review_session(session) is True
        loaded = store.get_review_session(session.session_id)
        assert loaded.status == ReviewStatus.COMPLETED
        assert loaded.thesis_status_after == ThesisStatus.STRENGTHENED

    def test_get_last_successful_review(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        fail = ReviewSession(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
            status=ReviewStatus.FAILED,
        )
        success = ReviewSession(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
            status=ReviewStatus.COMPLETED,
        )
        store.insert_review_session(fail)
        store.insert_review_session(success)
        last = store.get_last_successful_review("pos-1")
        assert last is not None
        assert last.status == ReviewStatus.COMPLETED

    def test_get_last_successful_review_excludes_fallback(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        fallback = ReviewSession(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
            status=ReviewStatus.COMPLETED,
            fallback_used=True,
        )
        store.insert_review_session(fallback)
        last = store.get_last_successful_review("pos-1")
        assert last is None

    def test_get_reviews_for_position(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        for _ in range(3):
            s = ReviewSession(
                position_id="pos-1",
                memory_id="mem-1",
                trigger_type=ReviewTriggerType.MANUAL,
            )
            store.insert_review_session(s)
        reviews = store.get_reviews_for_position("pos-1", limit=10)
        assert len(reviews) == 3


class TestReviewSchedule:
    def test_insert_and_due(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        past = datetime(2020, 1, 1)
        schedule = ReviewSchedule(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
            next_due_at=past,
        )
        assert store.insert_review_schedule(schedule) is True
        due = store.get_due_schedules()
        assert len(due) == 1
        assert due[0].schedule_id == schedule.schedule_id

    def test_get_due_excludes_future(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        future = datetime(2099, 1, 1)
        schedule = ReviewSchedule(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
            next_due_at=future,
        )
        store.insert_review_schedule(schedule)
        due = store.get_due_schedules()
        assert len(due) == 0

    def test_update_schedule_status(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        schedule = ReviewSchedule(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
            next_due_at=datetime.utcnow(),
        )
        store.insert_review_schedule(schedule)
        assert store.update_schedule_status(schedule.schedule_id, "SUPPRESSED") is True

    def test_get_schedules_for_position(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        s1 = ReviewSchedule(
            position_id="pos-1", memory_id="mem-1", trigger_type=ReviewTriggerType.MANUAL, next_due_at=datetime.utcnow()
        )
        s2 = ReviewSchedule(
            position_id="pos-1", memory_id="mem-1", trigger_type=ReviewTriggerType.PERIODIC_INTERVAL, next_due_at=datetime.utcnow()
        )
        store.insert_review_schedule(s1)
        store.insert_review_schedule(s2)
        schedules = store.get_schedules_for_position("pos-1")
        assert len(schedules) == 2


class TestTriggerEvent:
    def test_insert_and_get(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        trigger = ReviewTrigger(
            position_id="pos-1",
            trigger_type=TriggerType.PRICE_MOVEMENT,
            trigger_reason="Price hit 51000",
            trigger_value=51000.0,
        )
        assert store.insert_trigger_event(trigger) is True
        triggers = store.get_triggers_for_position("pos-1")
        assert len(triggers) == 1

    def test_suppress_trigger(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        trigger = ReviewTrigger(
            position_id="pos-1",
            trigger_type=TriggerType.PRICE_MOVEMENT,
        )
        store.insert_trigger_event(trigger)
        assert store.suppress_trigger(trigger.trigger_id) is True


class TestIntentExecution:
    def test_insert_and_pending(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        rec = IntentExecutionRecord(
            intent_id="int-1",
            position_id="pos-1",
            intent_type=IntentType.HOLD,
        )
        assert store.insert_intent_execution(rec) is True
        pending = store.get_pending_intents("pos-1")
        assert len(pending) == 1

    def test_update(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        rec = IntentExecutionRecord(
            intent_id="int-1",
            position_id="pos-1",
            intent_type=IntentType.HOLD,
        )
        store.insert_intent_execution(rec)
        rec.status = IntentStatus.EXECUTED
        rec.client_order_id = "TIM_HOLD_abc_1"
        assert store.update_intent_execution(rec) is True
        loaded = store.get_intent_execution(rec.execution_id)
        assert loaded.status == IntentStatus.EXECUTED
        assert loaded.client_order_id == "TIM_HOLD_abc_1"

    def test_pending_excludes_executed(self, store, sample_origin, sample_working):
        store.insert_origin(sample_origin)
        store.upsert_working_memory(sample_working)
        rec = IntentExecutionRecord(
            intent_id="int-1",
            position_id="pos-1",
            intent_type=IntentType.HOLD,
            status=IntentStatus.EXECUTED,
        )
        store.insert_intent_execution(rec)
        pending = store.get_pending_intents("pos-1")
        assert len(pending) == 0


class TestFeatureFlags:
    def test_default_false(self, store):
        assert store.get_feature_flag("test_flag") is False

    def test_set_and_get(self, store):
        store.set_feature_flag("test_flag", True)
        assert store.get_feature_flag("test_flag") is True

    def test_all_flags(self, store):
        store.set_feature_flag("flag_a", True)
        store.set_feature_flag("flag_b", False)
        all_flags = store.get_all_feature_flags()
        assert all_flags == {"flag_a": True, "flag_b": False}


class TestConfig:
    def test_default_none(self, store):
        assert store.get_config_value("nonexistent") is None

    def test_set_and_get(self, store):
        store.set_config_value("test_key", "test_value")
        assert store.get_config_value("test_key") == "test_value"

    def test_all_config(self, store):
        store.set_config_value("key_a", "val_a")
        store.set_config_value("key_b", "val_b")
        cfg = store.get_all_config()
        assert cfg == {"key_a": "val_a", "key_b": "val_b"}

    def test_load_tim_config_defaults(self, store):
        config = store.load_tim_config()
        assert config.tim_mode == TIMMode.OFF
        assert config.watchdog_timeout_minutes == 60

    def test_save_and_load_tim_config(self, store):
        config = TIMConfig(
            tim_mode=TIMMode.SHADOW,
            watchdog_timeout_minutes=30,
        )
        assert store.save_tim_config(config) is True
        loaded = store.load_tim_config()
        assert loaded.tim_mode == TIMMode.SHADOW
        assert loaded.watchdog_timeout_minutes == 30

    def test_save_invalid_mode(self, store):
        store.set_config_value("tim_mode", "INVALID")
        config = store.load_tim_config()
        assert config.tim_mode == TIMMode.OFF


class TestFaultTolerance:
    def test_create_schema_failure_does_not_raise(self, conn):
        t = TimStore(conn)
        conn.close()
        result = t.create_schema()
        assert result is False

    def test_schema_failure_disables_availability(self, conn):
        t = TimStore(conn)
        conn.close()
        t.create_schema()
        assert t.is_available() is False
