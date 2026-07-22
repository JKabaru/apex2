from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import duckdb
import structlog

from src.models.tim.config import TIMConfig
from src.models.tim.enums import TIMMode
from src.models.tim.intent import IntentExecutionRecord
from src.models.tim.review import ReviewSchedule, ReviewSession, ReviewTrigger
from src.models.tim.trade_memory import (
    JournalCompressionSummary,
    TradeJournalEntry,
    TradeMemoryRecoveryRecord,
    TradeOrigin,
    WorkingMemory,
)

logger = structlog.get_logger("tim_store")

TIM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trade_memory_origin (
    memory_id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    origin_episode_id VARCHAR NOT NULL,
    origin_quality VARCHAR NOT NULL DEFAULT 'UNKNOWN',
    entry_thesis VARCHAR NOT NULL DEFAULT '',
    entry_price DOUBLE NOT NULL DEFAULT 0.0,
    entry_atr DOUBLE,
    entry_timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    anchor_symbol VARCHAR NOT NULL DEFAULT '',
    timeframe VARCHAR NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL,
    metadata VARCHAR DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS trade_memory_working (
    memory_id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    checksum VARCHAR NOT NULL DEFAULT '',
    thesis_status VARCHAR NOT NULL DEFAULT 'INTACT',
    protection_mode VARCHAR NOT NULL DEFAULT 'MECHANICAL_ONLY',
    current_stop DOUBLE,
    current_target DOUBLE,
    unrealized_pnl_pct DOUBLE,
    mae_atr_multiple DOUBLE,
    mfe_atr_multiple DOUBLE,
    review_count INTEGER NOT NULL DEFAULT 0,
    failed_review_count INTEGER NOT NULL DEFAULT 0,
    last_review_timestamp TIMESTAMP,
    next_review_conditions VARCHAR DEFAULT '{}',
    watchdog_timer_start TIMESTAMP,
    watchdog_deadline TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    metadata VARCHAR DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS trade_memory_journal (
    journal_id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    memory_id VARCHAR NOT NULL,
    version INTEGER NOT NULL,
    event_type VARCHAR NOT NULL,
    event_data VARCHAR DEFAULT '{}',
    previous_checksum VARCHAR,
    new_checksum VARCHAR,
    timestamp TIMESTAMP NOT NULL,
    correlation_id VARCHAR
);

CREATE TABLE IF NOT EXISTS trade_memory_compression (
    compression_id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    memory_id VARCHAR NOT NULL,
    version_start INTEGER NOT NULL,
    version_end INTEGER NOT NULL,
    summary_data VARCHAR DEFAULT '{}',
    original_entry_count INTEGER NOT NULL,
    compressed_at TIMESTAMP NOT NULL,
    compression_version VARCHAR NOT NULL DEFAULT '1.0'
);

CREATE TABLE IF NOT EXISTS tim_review_session (
    session_id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    memory_id VARCHAR NOT NULL,
    trigger_type VARCHAR NOT NULL,
    trigger_id VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'PENDING',
    thesis_status_before VARCHAR,
    thesis_status_after VARCHAR,
    protection_mode_before VARCHAR NOT NULL DEFAULT 'MECHANICAL_ONLY',
    protection_mode_after VARCHAR,
    intent_envelope_id VARCHAR,
    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
    error_message VARCHAR,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    conditions_snapshot VARCHAR DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tim_review_schedule (
    schedule_id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    memory_id VARCHAR NOT NULL,
    trigger_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'ACTIVE',
    conditions VARCHAR DEFAULT '{}',
    interval_minutes DOUBLE,
    next_due_at TIMESTAMP NOT NULL,
    last_fired_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS tim_trigger_event (
    trigger_id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    trigger_type VARCHAR NOT NULL,
    trigger_reason VARCHAR NOT NULL DEFAULT '',
    trigger_value DOUBLE,
    triggered_at TIMESTAMP NOT NULL,
    schedule_id VARCHAR,
    suppressed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS tim_intent_execution (
    execution_id VARCHAR PRIMARY KEY,
    intent_id VARCHAR NOT NULL,
    position_id VARCHAR NOT NULL,
    intent_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'QUEUED',
    attempt_number INTEGER NOT NULL DEFAULT 1,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    client_order_id VARCHAR,
    stop_client_order_id VARCHAR,
    tp_client_order_id VARCHAR,
    exchange_order_id VARCHAR,
    error_message VARCHAR,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS tim_feature_flag_state (
    flag_name VARCHAR PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMP NOT NULL,
    updated_by VARCHAR DEFAULT 'SYSTEM'
);

CREATE TABLE IF NOT EXISTS tim_config_state (
    config_key VARCHAR PRIMARY KEY,
    config_value VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    updated_by VARCHAR DEFAULT 'SYSTEM'
);
"""


class TimStore:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self._conn = connection
        self.log = logger

    # ── Schema ────────────────────────────────────────────────────────

    def create_schema(self) -> bool:
        try:
            for statement in TIM_SCHEMA_SQL.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    self._conn.execute(stmt)
            self.log.info("TIM schema initialized")
            return True
        except Exception as exc:
            self.log.error("TIM schema creation failed", error=str(exc))
            return False

    def is_available(self) -> bool:
        required = [
            "trade_memory_origin",
            "trade_memory_working",
            "trade_memory_journal",
            "trade_memory_compression",
            "tim_review_session",
            "tim_review_schedule",
            "tim_trigger_event",
            "tim_intent_execution",
            "tim_feature_flag_state",
            "tim_config_state",
        ]
        try:
            existing = set(
                row[0]
                for row in self._conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
            )
            return all(t in existing for t in required)
        except Exception:
            return False

    def assert_available(self) -> None:
        if not self.is_available():
            raise RuntimeError("TIM schema is not fully available")

    # ── Trade Origin ──────────────────────────────────────────────────

    def insert_origin(self, origin: TradeOrigin) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO trade_memory_origin (
                    memory_id, position_id, origin_episode_id, origin_quality,
                    entry_thesis, entry_price, entry_atr, entry_timestamp,
                    symbol, side, anchor_symbol, timeframe, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    origin.memory_id,
                    origin.position_id,
                    origin.origin_episode_id,
                    origin.origin_quality.value,
                    origin.entry_thesis,
                    origin.entry_price,
                    origin.entry_atr,
                    origin.entry_timestamp,
                    origin.symbol,
                    origin.side,
                    origin.anchor_symbol,
                    origin.timeframe,
                    origin.created_at,
                    json.dumps(origin.metadata),
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("insert_origin failed", error=str(exc))
            return False

    def get_origin_by_position(self, position_id: str) -> Optional[TradeOrigin]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_origin WHERE position_id = ?",
            [position_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_origin(row, cols)

    def get_origin_by_episode(self, episode_id: str) -> Optional[TradeOrigin]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_origin WHERE origin_episode_id = ?",
            [episode_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_origin(row, cols)

    @staticmethod
    def _row_to_origin(row, cols) -> TradeOrigin:
        data = dict(zip(cols, row))
        return TradeOrigin(
            memory_id=data["memory_id"],
            position_id=data["position_id"],
            origin_episode_id=data["origin_episode_id"],
            origin_quality=data["origin_quality"],
            entry_thesis=data["entry_thesis"],
            entry_price=data["entry_price"],
            entry_atr=data["entry_atr"],
            entry_timestamp=data["entry_timestamp"],
            symbol=data["symbol"],
            side=data["side"],
            anchor_symbol=data["anchor_symbol"],
            timeframe=data["timeframe"],
            created_at=data["created_at"],
            metadata=json.loads(data["metadata"]) if isinstance(data.get("metadata"), str) else {},
        )

    # ── Working Memory ───────────────────────────────────────────────

    def upsert_working_memory(self, memory: WorkingMemory) -> bool:
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO trade_memory_working (
                    memory_id, position_id, version, checksum,
                    thesis_status, protection_mode,
                    current_stop, current_target,
                    unrealized_pnl_pct, mae_atr_multiple, mfe_atr_multiple,
                    review_count, failed_review_count, last_review_timestamp,
                    next_review_conditions,
                    watchdog_timer_start, watchdog_deadline,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    memory.memory_id,
                    memory.position_id,
                    memory.version,
                    memory.checksum,
                    memory.thesis_status.value,
                    memory.protection_mode.value,
                    memory.current_stop,
                    memory.current_target,
                    memory.unrealized_pnl_pct,
                    memory.mae_atr_multiple,
                    memory.mfe_atr_multiple,
                    memory.review_count,
                    memory.failed_review_count,
                    memory.last_review_timestamp,
                    memory.next_review_conditions,
                    memory.watchdog_timer_start,
                    memory.watchdog_deadline,
                    memory.created_at,
                    memory.updated_at,
                    json.dumps(memory.metadata),
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("upsert_working_memory failed", error=str(exc))
            return False

    def get_working_memory(self, memory_id: str) -> Optional[WorkingMemory]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_working WHERE memory_id = ?",
            [memory_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_working(row, cols)

    def get_working_memory_by_position(self, position_id: str) -> Optional[WorkingMemory]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_working WHERE position_id = ?",
            [position_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_working(row, cols)

    def get_working_memory_version(self, memory_id: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT version FROM trade_memory_working WHERE memory_id = ?",
            [memory_id],
        ).fetchone()
        return row[0] if row else None

    def get_working_memory_checksum(self, memory_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT checksum FROM trade_memory_working WHERE memory_id = ?",
            [memory_id],
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _row_to_working(row, cols) -> WorkingMemory:
        data = dict(zip(cols, row))
        return WorkingMemory(
            memory_id=data["memory_id"],
            position_id=data["position_id"],
            version=data["version"],
            checksum=data["checksum"],
            thesis_status=data["thesis_status"],
            protection_mode=data["protection_mode"],
            current_stop=data["current_stop"],
            current_target=data["current_target"],
            unrealized_pnl_pct=data["unrealized_pnl_pct"],
            mae_atr_multiple=data["mae_atr_multiple"],
            mfe_atr_multiple=data["mfe_atr_multiple"],
            review_count=data["review_count"],
            failed_review_count=data["failed_review_count"],
            last_review_timestamp=data["last_review_timestamp"],
            next_review_conditions=data["next_review_conditions"],
            watchdog_timer_start=data["watchdog_timer_start"],
            watchdog_deadline=data["watchdog_deadline"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=json.loads(data["metadata"]) if isinstance(data.get("metadata"), str) else {},
        )

    # ── Journal ───────────────────────────────────────────────────────

    def append_journal_entry(self, entry: TradeJournalEntry) -> bool:
        try:
            latest = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM trade_memory_journal WHERE position_id = ?",
                [entry.position_id],
            ).fetchone()[0]
            if entry.version != latest + 1:
                self.log.warning(
                    "journal version not monotonic",
                    expected=latest + 1,
                    got=entry.version,
                )
                return False
            self._conn.execute(
                """
                INSERT INTO trade_memory_journal (
                    journal_id, position_id, memory_id, version, event_type,
                    event_data, previous_checksum, new_checksum,
                    timestamp, correlation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    entry.journal_id,
                    entry.position_id,
                    entry.memory_id,
                    entry.version,
                    entry.event_type.value,
                    entry.event_data,
                    entry.previous_checksum,
                    entry.new_checksum,
                    entry.timestamp,
                    entry.correlation_id,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("append_journal_entry failed", error=str(exc))
            return False

    def get_journal_entries(self, position_id: str, limit: int = 100) -> list[TradeJournalEntry]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_journal WHERE position_id = ? ORDER BY version DESC LIMIT ?",
            [position_id, limit],
        )
        cols = [desc[0] for desc in result.description]
        return [self._row_to_journal(r, cols) for r in result.fetchall()]

    def get_journal_entries_after_version(
        self, position_id: str, version: int
    ) -> list[TradeJournalEntry]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_journal WHERE position_id = ? AND version > ? ORDER BY version",
            [position_id, version],
        )
        cols = [desc[0] for desc in result.description]
        return [self._row_to_journal(r, cols) for r in result.fetchall()]

    def get_journal_entry_count(self, position_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM trade_memory_journal WHERE position_id = ?",
            [position_id],
        ).fetchone()
        return row[0] if row else 0

    def get_latest_journal_version(self, position_id: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT MAX(version) FROM trade_memory_journal WHERE position_id = ?",
            [position_id],
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _row_to_journal(row, cols) -> TradeJournalEntry:
        data = dict(zip(cols, row))
        return TradeJournalEntry(
            journal_id=data["journal_id"],
            position_id=data["position_id"],
            memory_id=data["memory_id"],
            version=data["version"],
            event_type=data["event_type"],
            event_data=data["event_data"],
            previous_checksum=data["previous_checksum"],
            new_checksum=data["new_checksum"],
            timestamp=data["timestamp"],
            correlation_id=data["correlation_id"],
        )

    # ── Compression ───────────────────────────────────────────────────

    def insert_compression_summary(self, summary: JournalCompressionSummary) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO trade_memory_compression (
                    compression_id, position_id, memory_id,
                    version_start, version_end, summary_data,
                    original_entry_count, compressed_at, compression_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    summary.compression_id,
                    summary.position_id,
                    summary.memory_id,
                    summary.version_start,
                    summary.version_end,
                    summary.summary_data,
                    summary.original_entry_count,
                    summary.compressed_at,
                    summary.compression_version,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("insert_compression_summary failed", error=str(exc))
            return False

    def get_latest_compression(self, position_id: str) -> Optional[JournalCompressionSummary]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_compression WHERE position_id = ? ORDER BY version_end DESC LIMIT 1",
            [position_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_compression(row, cols)

    def get_compression_for_version(
        self, position_id: str, version: int
    ) -> Optional[JournalCompressionSummary]:
        result = self._conn.execute(
            "SELECT * FROM trade_memory_compression WHERE position_id = ? AND version_start <= ? AND version_end >= ?",
            [position_id, version, version],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_compression(row, cols)

    @staticmethod
    def _row_to_compression(row, cols) -> JournalCompressionSummary:
        data = dict(zip(cols, row))
        return JournalCompressionSummary(
            compression_id=data["compression_id"],
            position_id=data["position_id"],
            memory_id=data["memory_id"],
            version_start=data["version_start"],
            version_end=data["version_end"],
            summary_data=data["summary_data"],
            original_entry_count=data["original_entry_count"],
            compressed_at=data["compressed_at"],
            compression_version=data["compression_version"],
        )

    # ── Review Session ────────────────────────────────────────────────

    def insert_review_session(self, session: ReviewSession) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO tim_review_session (
                    session_id, position_id, memory_id, trigger_type, trigger_id,
                    status, thesis_status_before, thesis_status_after,
                    protection_mode_before, protection_mode_after,
                    intent_envelope_id, fallback_used, error_message,
                    started_at, completed_at, conditions_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    session.session_id,
                    session.position_id,
                    session.memory_id,
                    session.trigger_type.value,
                    session.trigger_id,
                    session.status.value,
                    session.thesis_status_before.value if session.thesis_status_before else None,
                    session.thesis_status_after.value if session.thesis_status_after else None,
                    session.protection_mode_before.value,
                    session.protection_mode_after.value if session.protection_mode_after else None,
                    session.intent_envelope_id,
                    session.fallback_used,
                    session.error_message,
                    session.started_at,
                    session.completed_at,
                    session.conditions_snapshot,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("insert_review_session failed", error=str(exc))
            return False

    def update_review_session(self, session: ReviewSession) -> bool:
        try:
            self._conn.execute(
                """
                UPDATE tim_review_session SET
                    status = ?,
                    thesis_status_after = ?,
                    protection_mode_after = ?,
                    intent_envelope_id = ?,
                    fallback_used = ?,
                    error_message = ?,
                    completed_at = ?,
                    conditions_snapshot = ?
                WHERE session_id = ?
                """,
                [
                    session.status.value,
                    session.thesis_status_after.value if session.thesis_status_after else None,
                    session.protection_mode_after.value if session.protection_mode_after else None,
                    session.intent_envelope_id,
                    session.fallback_used,
                    session.error_message,
                    session.completed_at,
                    session.conditions_snapshot,
                    session.session_id,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("update_review_session failed", error=str(exc))
            return False

    def get_review_session(self, session_id: str) -> Optional[ReviewSession]:
        result = self._conn.execute(
            "SELECT * FROM tim_review_session WHERE session_id = ?",
            [session_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_review_session(row, cols)

    def get_reviews_for_position(self, position_id: str, limit: int = 10) -> list[ReviewSession]:
        result = self._conn.execute(
            "SELECT * FROM tim_review_session WHERE position_id = ? ORDER BY started_at DESC LIMIT ?",
            [position_id, limit],
        )
        cols = [desc[0] for desc in result.description]
        return [self._row_to_review_session(r, cols) for r in result.fetchall()]

    def get_last_successful_review(self, position_id: str) -> Optional[ReviewSession]:
        result = self._conn.execute(
            """
            SELECT * FROM tim_review_session
            WHERE position_id = ?
              AND status IN ('COMPLETED', 'EXECUTED')
              AND (fallback_used = FALSE OR fallback_used IS NULL)
            ORDER BY started_at DESC LIMIT 1
            """,
            [position_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_review_session(row, cols)

    @staticmethod
    def _row_to_review_session(row, cols) -> ReviewSession:
        data = dict(zip(cols, row))
        return ReviewSession(
            session_id=data["session_id"],
            position_id=data["position_id"],
            memory_id=data["memory_id"],
            trigger_type=data["trigger_type"],
            trigger_id=data["trigger_id"],
            status=data["status"],
            thesis_status_before=data["thesis_status_before"],
            thesis_status_after=data["thesis_status_after"],
            protection_mode_before=data["protection_mode_before"],
            protection_mode_after=data["protection_mode_after"],
            intent_envelope_id=data["intent_envelope_id"],
            fallback_used=data["fallback_used"],
            error_message=data["error_message"],
            started_at=data["started_at"],
            completed_at=data["completed_at"],
            conditions_snapshot=data["conditions_snapshot"],
        )

    # ── Review Schedule ───────────────────────────────────────────────

    def insert_review_schedule(self, schedule: ReviewSchedule) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO tim_review_schedule (
                    schedule_id, position_id, memory_id, trigger_type,
                    status, conditions, interval_minutes,
                    next_due_at, last_fired_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    schedule.schedule_id,
                    schedule.position_id,
                    schedule.memory_id,
                    schedule.trigger_type.value,
                    schedule.status,
                    schedule.conditions,
                    schedule.interval_minutes,
                    schedule.next_due_at,
                    schedule.last_fired_at,
                    schedule.created_at,
                    schedule.updated_at,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("insert_review_schedule failed", error=str(exc))
            return False

    def get_due_schedules(self, limit: int = 50) -> list[ReviewSchedule]:
        now = datetime.utcnow()
        result = self._conn.execute(
            "SELECT * FROM tim_review_schedule WHERE status = 'ACTIVE' AND next_due_at <= ? ORDER BY next_due_at LIMIT ?",
            [now, limit],
        )
        cols = [desc[0] for desc in result.description]
        return [self._row_to_schedule(r, cols) for r in result.fetchall()]

    def get_schedules_for_position(self, position_id: str) -> list[ReviewSchedule]:
        result = self._conn.execute(
            "SELECT * FROM tim_review_schedule WHERE position_id = ?",
            [position_id],
        )
        cols = [desc[0] for desc in result.description]
        return [self._row_to_schedule(r, cols) for r in result.fetchall()]

    def update_schedule_status(self, schedule_id: str, status: str) -> bool:
        try:
            self._conn.execute(
                "UPDATE tim_review_schedule SET status = ?, updated_at = ? WHERE schedule_id = ?",
                [status, datetime.utcnow(), schedule_id],
            )
            return True
        except Exception as exc:
            self.log.warning("update_schedule_status failed", error=str(exc))
            return False

    def update_schedule_next_due(self, schedule_id: str, next_due: datetime) -> bool:
        try:
            self._conn.execute(
                "UPDATE tim_review_schedule SET next_due_at = ?, updated_at = ? WHERE schedule_id = ?",
                [next_due, datetime.utcnow(), schedule_id],
            )
            return True
        except Exception as exc:
            self.log.warning("update_schedule_next_due failed", error=str(exc))
            return False

    def update_schedule_conditions(self, schedule_id: str, conditions: str) -> bool:
        try:
            self._conn.execute(
                "UPDATE tim_review_schedule SET conditions = ?, updated_at = ? WHERE schedule_id = ?",
                [conditions, datetime.utcnow(), schedule_id],
            )
            return True
        except Exception as exc:
            self.log.warning("update_schedule_conditions failed", error=str(exc))
            return False

    @staticmethod
    def _row_to_schedule(row, cols) -> ReviewSchedule:
        data = dict(zip(cols, row))
        return ReviewSchedule(
            schedule_id=data["schedule_id"],
            position_id=data["position_id"],
            memory_id=data["memory_id"],
            trigger_type=data["trigger_type"],
            status=data["status"],
            conditions=data["conditions"],
            interval_minutes=data["interval_minutes"],
            next_due_at=data["next_due_at"],
            last_fired_at=data["last_fired_at"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    # ── Trigger Event ─────────────────────────────────────────────────

    def insert_trigger_event(self, trigger: ReviewTrigger) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO tim_trigger_event (
                    trigger_id, position_id, trigger_type, trigger_reason,
                    trigger_value, triggered_at, schedule_id, suppressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    trigger.trigger_id,
                    trigger.position_id,
                    trigger.trigger_type.value,
                    trigger.trigger_reason,
                    trigger.trigger_value,
                    trigger.triggered_at,
                    trigger.schedule_id,
                    trigger.suppressed,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("insert_trigger_event failed", error=str(exc))
            return False

    def get_triggers_for_position(self, position_id: str, limit: int = 50) -> list[ReviewTrigger]:
        result = self._conn.execute(
            "SELECT * FROM tim_trigger_event WHERE position_id = ? ORDER BY triggered_at DESC LIMIT ?",
            [position_id, limit],
        )
        cols = [desc[0] for desc in result.description]
        return [self._row_to_trigger(r, cols) for r in result.fetchall()]

    def suppress_trigger(self, trigger_id: str) -> bool:
        try:
            self._conn.execute(
                "UPDATE tim_trigger_event SET suppressed = TRUE WHERE trigger_id = ?",
                [trigger_id],
            )
            return True
        except Exception as exc:
            self.log.warning("suppress_trigger failed", error=str(exc))
            return False

    @staticmethod
    def _row_to_trigger(row, cols) -> ReviewTrigger:
        data = dict(zip(cols, row))
        return ReviewTrigger(
            trigger_id=data["trigger_id"],
            position_id=data["position_id"],
            trigger_type=data["trigger_type"],
            trigger_reason=data["trigger_reason"],
            trigger_value=data["trigger_value"],
            triggered_at=data["triggered_at"],
            schedule_id=data["schedule_id"],
            suppressed=data["suppressed"],
        )

    # ── Intent Execution ──────────────────────────────────────────────

    def insert_intent_execution(self, record: IntentExecutionRecord) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO tim_intent_execution (
                    execution_id, intent_id, position_id, intent_type,
                    status, attempt_number, max_attempts,
                    client_order_id, stop_client_order_id, tp_client_order_id,
                    exchange_order_id, error_message,
                    started_at, completed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.execution_id,
                    record.intent_id,
                    record.position_id,
                    record.intent_type.value,
                    record.status.value,
                    record.attempt_number,
                    record.max_attempts,
                    record.client_order_id,
                    record.stop_client_order_id,
                    record.tp_client_order_id,
                    record.exchange_order_id,
                    record.error_message,
                    record.started_at,
                    record.completed_at,
                    record.created_at,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("insert_intent_execution failed", error=str(exc))
            return False

    def update_intent_execution(self, record: IntentExecutionRecord) -> bool:
        try:
            self._conn.execute(
                """
                UPDATE tim_intent_execution SET
                    status = ?,
                    client_order_id = ?,
                    stop_client_order_id = ?,
                    tp_client_order_id = ?,
                    exchange_order_id = ?,
                    error_message = ?,
                    started_at = ?,
                    completed_at = ?
                WHERE execution_id = ?
                """,
                [
                    record.status.value,
                    record.client_order_id,
                    record.stop_client_order_id,
                    record.tp_client_order_id,
                    record.exchange_order_id,
                    record.error_message,
                    record.started_at,
                    record.completed_at,
                    record.execution_id,
                ],
            )
            return True
        except Exception as exc:
            self.log.warning("update_intent_execution failed", error=str(exc))
            return False

    def get_intent_execution(self, execution_id: str) -> Optional[IntentExecutionRecord]:
        result = self._conn.execute(
            "SELECT * FROM tim_intent_execution WHERE execution_id = ?",
            [execution_id],
        )
        row = result.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in result.description]
        return self._row_to_intent_execution(row, cols)

    def get_pending_intents(self, position_id: str) -> list[IntentExecutionRecord]:
        result = self._conn.execute(
            """
            SELECT * FROM tim_intent_execution
            WHERE position_id = ?
              AND status IN ('PROPOSED', 'QUEUED', 'EXECUTING')
            ORDER BY created_at
            """,
            [position_id],
        )
        cols = [desc[0] for desc in result.description]
        return [self._row_to_intent_execution(r, cols) for r in result.fetchall()]

    @staticmethod
    def _row_to_intent_execution(row, cols) -> IntentExecutionRecord:
        data = dict(zip(cols, row))
        return IntentExecutionRecord(
            execution_id=data["execution_id"],
            intent_id=data["intent_id"],
            position_id=data["position_id"],
            intent_type=data["intent_type"],
            status=data["status"],
            attempt_number=data["attempt_number"],
            max_attempts=data["max_attempts"],
            client_order_id=data["client_order_id"],
            stop_client_order_id=data["stop_client_order_id"],
            tp_client_order_id=data["tp_client_order_id"],
            exchange_order_id=data["exchange_order_id"],
            error_message=data["error_message"],
            started_at=data["started_at"],
            completed_at=data["completed_at"],
            created_at=data["created_at"],
        )

    # ── Feature Flag ──────────────────────────────────────────────────

    def get_feature_flag(self, flag_name: str) -> bool:
        row = self._conn.execute(
            "SELECT enabled FROM tim_feature_flag_state WHERE flag_name = ?",
            [flag_name],
        ).fetchone()
        return row[0] if row else False

    def set_feature_flag(self, flag_name: str, enabled: bool, updated_by: str = "SYSTEM") -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tim_feature_flag_state (flag_name, enabled, updated_at, updated_by)
            VALUES (?, ?, ?, ?)
            """,
            [flag_name, enabled, datetime.utcnow(), updated_by],
        )

    def get_all_feature_flags(self) -> dict[str, bool]:
        rows = self._conn.execute("SELECT flag_name, enabled FROM tim_feature_flag_state").fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Config State ──────────────────────────────────────────────────

    def get_config_value(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT config_value FROM tim_config_state WHERE config_key = ?",
            [key],
        ).fetchone()
        return row[0] if row else None

    def set_config_value(self, key: str, value: str, updated_by: str = "SYSTEM") -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tim_config_state (config_key, config_value, updated_at, updated_by)
            VALUES (?, ?, ?, ?)
            """,
            [key, value, datetime.utcnow(), updated_by],
        )

    def get_all_config(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT config_key, config_value FROM tim_config_state").fetchall()
        return {r[0]: r[1] for r in rows}

    def load_tim_config(self) -> TIMConfig:
        raw = self.get_all_config()
        mode_str = raw.get("tim_mode", "OFF")
        try:
            tim_mode = TIMMode(mode_str)
        except ValueError:
            tim_mode = TIMMode.OFF
        return TIMConfig(
            tim_mode=tim_mode,
            watchdog_timeout_minutes=int(raw.get("watchdog_timeout_minutes", 60)),
            max_intent_retries=int(raw.get("max_intent_retries", 3)),
            default_review_interval_minutes=int(raw.get("default_review_interval_minutes", 240)),
            max_journal_entries_before_compression=int(
                raw.get("max_journal_entries_before_compression", 500)
            ),
            prompt_version=raw.get("prompt_version", "1.0"),
            schema_version=raw.get("schema_version", "1.0"),
            config_version=raw.get("config_version", "1.0"),
        )

    def save_tim_config(self, config: TIMConfig) -> bool:
        try:
            if not isinstance(config.tim_mode, TIMMode):
                return False
            pairs = [
                ("tim_mode", config.tim_mode.value),
                ("watchdog_timeout_minutes", str(config.watchdog_timeout_minutes)),
                ("max_intent_retries", str(config.max_intent_retries)),
                ("default_review_interval_minutes", str(config.default_review_interval_minutes)),
                (
                    "max_journal_entries_before_compression",
                    str(config.max_journal_entries_before_compression),
                ),
                ("prompt_version", config.prompt_version),
                ("schema_version", config.schema_version),
                ("config_version", config.config_version),
            ]
            for key, value in pairs:
                self.set_config_value(key, value)
            return True
        except Exception as exc:
            self.log.warning("save_tim_config failed", error=str(exc))
            return False
