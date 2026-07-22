from __future__ import annotations

import json
from datetime import datetime

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
    TradeMemoryRecoveryState,
    TriggerType,
)
from src.models.tim.intent import IntentExecutionRecord, StrategicIntentEnvelope, TradeManagementIntent
from src.models.tim.review import (
    ReviewConditions,
    ReviewSchedule,
    ReviewSession,
    ReviewTrigger,
    TriggerSuppressionRecord,
)
from src.models.tim.trade_memory import (
    JournalCompressionSummary,
    TradeJournalEntry,
    TradeMemoryRecoveryRecord,
    TradeOrigin,
    WorkingMemory,
)


class TestModelDefaults:
    def test_trade_origin_minimal(self):
        origin = TradeOrigin(position_id="pos-1", origin_episode_id="ep-1", symbol="BTCUSDT", side="LONG")
        assert origin.position_id == "pos-1"
        assert origin.origin_quality == OriginQuality.UNKNOWN
        assert origin.entry_price == 0.0
        assert origin.metadata == {}

    def test_working_memory_defaults(self):
        wm = WorkingMemory(memory_id="mem-1", position_id="pos-1")
        assert wm.version == 1
        assert wm.thesis_status == ThesisStatus.INTACT
        assert wm.protection_mode == ProtectionMode.MECHANICAL_ONLY
        assert wm.review_count == 0
        assert wm.failed_review_count == 0
        assert wm.next_review_conditions == "{}"

    def test_trade_journal_entry_defaults(self):
        entry = TradeJournalEntry(
            position_id="pos-1",
            memory_id="mem-1",
            version=1,
            event_type=JournalEventType.SYSTEM_CREATED,
        )
        assert entry.event_data == "{}"
        assert entry.previous_checksum is None
        assert entry.new_checksum is None

    def test_intent_execution_record_defaults(self):
        rec = IntentExecutionRecord(intent_id="int-1", position_id="pos-1", intent_type=IntentType.HOLD)
        assert rec.status == IntentStatus.QUEUED
        assert rec.attempt_number == 1
        assert rec.max_attempts == 3
        assert rec.client_order_id is None
        assert rec.stop_client_order_id is None
        assert rec.tp_client_order_id is None

    def test_review_session_defaults(self):
        session = ReviewSession(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
        )
        assert session.status == ReviewStatus.PENDING
        assert session.protection_mode_before == ProtectionMode.MECHANICAL_ONLY
        assert session.fallback_used is False
        assert session.conditions_snapshot == "{}"

    def test_review_conditions_defaults(self):
        rc = ReviewConditions()
        assert rc.reference_price is None
        assert rc.reference_atr is None
        assert rc.reference_stop is None
        assert rc.reference_target is None
        assert rc.reference_trend is None
        assert rc.reference_volatility is None
        assert rc.min_price_delta_pct is None
        assert rc.max_time_delta_minutes is None

    def test_tim_config_defaults(self):
        config = TIMConfig()
        assert config.tim_mode == TIMMode.OFF
        assert config.watchdog_timeout_minutes == 60
        assert config.max_intent_retries == 3
        assert config.default_review_interval_minutes == 240
        assert config.max_journal_entries_before_compression == 500
        assert config.prompt_version == "1.0"
        assert config.schema_version == "1.0"
        assert config.config_version == "1.0"

    def test_review_schedule_defaults(self):
        schedule = ReviewSchedule(
            position_id="pos-1",
            memory_id="mem-1",
            trigger_type=ReviewTriggerType.MANUAL,
            next_due_at=datetime.utcnow(),
        )
        assert schedule.status == "ACTIVE"
        assert schedule.conditions == "{}"

    def test_trigger_suppression_defaults(self):
        record = TriggerSuppressionRecord(
            position_id="pos-1",
            trigger_id="trig-1",
            suppressed_by="SYSTEM",
        )
        assert record.reason == ""

    def test_trade_memory_recovery_defaults(self):
        record = TradeMemoryRecoveryRecord(position_id="pos-1")
        assert record.recovery_state == TradeMemoryRecoveryState.NOT_REQUIRED
        assert record.error_message is None

    def test_compression_summary_defaults(self):
        summary = JournalCompressionSummary(
            position_id="pos-1",
            memory_id="mem-1",
            version_start=1,
            version_end=10,
            original_entry_count=10,
        )
        assert summary.compression_version == "1.0"
        assert summary.summary_data == "{}"


class TestEnums:
    def test_thesis_status_terminal(self):
        assert ThesisStatus.INVALIDATED.is_terminal is True

    def test_thesis_status_non_terminal(self):
        for status in [ThesisStatus.INTACT, ThesisStatus.STRENGTHENED, ThesisStatus.EVOLVED, ThesisStatus.WEAKENED, ThesisStatus.AT_RISK]:
            assert status.is_terminal is False

    def test_protection_mode_values(self):
        assert set(ProtectionMode.__members__) == {"MECHANICAL_ONLY", "TIM_SUPERVISED", "DEGRADED_MECHANICAL"}

    def test_tim_mode_values(self):
        assert set(TIMMode.__members__) == {"OFF", "MEMORY_ONLY", "SHADOW", "LIVE_PROTECTION", "LIVE_FULL"}

    def test_origin_quality_values(self):
        assert set(OriginQuality.__members__) == {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}

    def test_journal_event_type_values(self):
        types = JournalEventType.__members__
        assert "ORIGIN_SET" in types
        assert "REVIEW_COMPLETED" in types
        assert "INTENT_EXECUTED" in types
        assert "MEMORY_COMPRESSED" in types
        assert "THESIS_UPDATED" in types
        assert "DEGRADATION_TRIGGERED" in types
        assert "SYSTEM_CREATED" in types

    def test_intent_type_values(self):
        types = IntentType.__members__
        assert "HOLD" in types
        assert "INCREASE_PROTECTION" in types
        assert "DECREASE_PROTECTION" in types
        assert "SCALE_OUT" in types
        assert "EXIT" in types
        assert "CANCEL_EXIT" in types
        assert "ADJUST_TARGET" in types
        assert "REQUEST_REVIEW" in types
        assert "ESCALATE" in types

    def test_review_status_values(self):
        assert set(ReviewStatus.__members__) == {"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "SKIPPED", "TIMEOUT"}

    def test_intent_status_values(self):
        assert set(IntentStatus.__members__) == {"PROPOSED", "QUEUED", "EXECUTING", "EXECUTED", "FAILED", "REJECTED", "SUPERSEDED"}

    def test_trigger_type_values(self):
        types = TriggerType.__members__
        assert "PRICE_MOVEMENT" in types
        assert "REGIME_CHANGE" in types
        assert "POSITION_CLOSED" in types

    def test_review_trigger_type_values(self):
        types = ReviewTriggerType.__members__
        assert "PRICE_THRESHOLD" in types
        assert "TIME_ELAPSED" in types
        assert "SYSTEM_STARTUP" in types


class TestReviewConditionsRoundtrip:
    def test_json_roundtrip(self):
        rc = ReviewConditions(
            reference_price=50000.0,
            reference_atr=1200.0,
            reference_stop=48000.0,
            reference_target=55000.0,
            reference_trend="BULLISH",
            reference_volatility="HIGH",
            min_price_delta_pct=2.5,
            max_time_delta_minutes=240.0,
        )
        dumped = json.loads(rc.model_dump_json())
        restored = ReviewConditions(**dumped)
        assert restored.reference_price == 50000.0
        assert restored.reference_atr == 1200.0
        assert restored.reference_trend == "BULLISH"
        assert restored.min_price_delta_pct == 2.5

    def test_empty_conditions_roundtrip(self):
        rc = ReviewConditions()
        dumped = json.loads(rc.model_dump_json())
        restored = ReviewConditions(**dumped)
        assert restored.reference_price is None


class TestIntentExecutionClientOrderId:
    def test_client_order_id_fields(self):
        rec = IntentExecutionRecord(
            intent_id="int-1",
            position_id="pos-1",
            intent_type=IntentType.EXIT,
            client_order_id="TIM_EXIT_abcd1234_1",
            stop_client_order_id="TIM_SL_abcd1234_1",
            tp_client_order_id="TIM_TP_abcd1234_1",
        )
        assert rec.client_order_id == "TIM_EXIT_abcd1234_1"
        assert rec.stop_client_order_id == "TIM_SL_abcd1234_1"
        assert rec.tp_client_order_id == "TIM_TP_abcd1234_1"

    def test_client_order_id_nullable(self):
        rec = IntentExecutionRecord(intent_id="int-1", position_id="pos-1", intent_type=IntentType.HOLD)
        assert rec.client_order_id is None


class TestThesisTransitions:
    def test_valid_transition_intact_to_strengthened(self):
        wm = WorkingMemory(memory_id="mem-1", position_id="pos-1", thesis_status=ThesisStatus.INTACT)
        wm.thesis_status = ThesisStatus.STRENGTHENED
        assert wm.thesis_status == ThesisStatus.STRENGTHENED

    def test_valid_transition_weakened_to_invalidated(self):
        wm = WorkingMemory(memory_id="mem-1", position_id="pos-1", thesis_status=ThesisStatus.WEAKENED)
        wm.thesis_status = ThesisStatus.INVALIDATED
        assert wm.thesis_status == ThesisStatus.INVALIDATED


class TestJournalVersion:
    def test_version_required(self):
        with pytest.raises(Exception):
            TradeJournalEntry(
                position_id="pos-1",
                memory_id="mem-1",
                version=0,
                event_type=JournalEventType.SYSTEM_CREATED,
            )

    def test_version_ge_one(self):
        entry = TradeJournalEntry(
            position_id="pos-1",
            memory_id="mem-1",
            version=1,
            event_type=JournalEventType.SYSTEM_CREATED,
        )
        assert entry.version == 1


class TestTIMConfigValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(Exception):
            TIMConfig(tim_mode="INVALID")

    def test_valid_mode_accepts(self):
        config = TIMConfig(tim_mode=TIMMode.SHADOW)
        assert config.tim_mode == TIMMode.SHADOW
        assert config.tim_mode.value == "SHADOW"


class TestWorkingMemoryChecksum:
    def test_checksum_required_when_version_gt_one(self):
        with pytest.raises(ValueError):
            WorkingMemory(memory_id="mem-1", position_id="pos-1", version=2, checksum="")

    def test_checksum_ok_when_version_one(self):
        wm = WorkingMemory(memory_id="mem-1", position_id="pos-1", version=1, checksum="")
        assert wm.version == 1


class TestStrategicIntentEnvelope:
    def test_envelope_with_intents(self):
        intent = TradeManagementIntent(
            position_id="pos-1",
            memory_id="mem-1",
            intent_type=IntentType.HOLD,
        )
        envelope = StrategicIntentEnvelope(
            position_id="pos-1",
            review_session_id="sess-1",
            intents=[intent],
        )
        assert len(envelope.intents) == 1
        assert envelope.intents[0].intent_type == IntentType.HOLD
