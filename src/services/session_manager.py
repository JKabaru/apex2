from __future__ import annotations

import json
import platform
from datetime import datetime
from typing import Optional

import duckdb
import structlog

from src.models.session import TradingSession

logger = structlog.get_logger("session_manager")

CONFIG_DB = "data/configuration_profiles.duckdb"


class SessionManager:
    """Manages trading session lifecycle.

    A session begins at startup (after profile selection) and ends at shutdown.
    Every trade executed during the session carries its session_id for traceability.
    """

    def __init__(self, db_path: str = CONFIG_DB):
        self._conn = duckdb.connect(db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_sessions (
                session_id VARCHAR PRIMARY KEY,
                profile_id VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP,
                payload_json JSON NOT NULL
            )
        """)

    def start_session(
        self,
        profile_id: str,
        git_commit: str = "",
        system_version: str = "",
        operator: str = "",
        startup_reason: str = "",
        config_hash: str = "",
    ) -> TradingSession:
        session = TradingSession(
            configuration_profile_id=profile_id,
            git_commit=git_commit,
            system_version=system_version,
            operator=operator,
            startup_reason=startup_reason,
            hostname=platform.node(),
            config_hash=config_hash,
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trading_sessions
                (session_id, profile_id, started_at, ended_at, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                session.session_id,
                profile_id,
                session.started_at.isoformat(),
                None,
                json.dumps(session.model_dump(mode="json")),
            ],
        )
        logger.info(
            "Trading session started",
            session_id=session.session_id,
            profile_id=profile_id,
            config_hash=config_hash,
        )
        return session

    def end_session(self, session_id: str) -> None:
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE trading_sessions SET ended_at = ? WHERE session_id = ?",
            [now, session_id],
        )
        logger.info("Trading session ended", session_id=session_id)

    def get_active_session(self) -> Optional[TradingSession]:
        row = self._conn.execute(
            "SELECT payload_json FROM trading_sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
        ).fetchone()
        if row is None:
            return None
        return TradingSession.model_validate(json.loads(row[0]))

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
