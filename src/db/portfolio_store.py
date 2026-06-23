from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import duckdb
import structlog

from src.core.models import Position, PositionState, SystemEvent

logger = structlog.get_logger("portfolio_store")

PORTFOLIO_DB = "data/apex_portfolio.duckdb"

POSITION_COLUMNS = [
    "position_id",
    "symbol",
    "side",
    "quantity",
    "avg_fill_price",
    "fees",
    "exchange_order_ids",
    "entry_timestamp",
    "exit_timestamp",
    "entry_thesis",
    "anchor_symbol",
    "correlation_score",
    "initial_stop_loss",
    "initial_take_profit",
    "current_stop",
    "current_target",
    "highest_unrealized_profit",
    "maximum_drawdown",
    "review_count",
    "current_recommendation",
    "lifecycle_state",
    "exit_reason",
]


class PortfolioStore:
    def __init__(self, db_path: str = PORTFOLIO_DB):
        self._conn = duckdb.connect(db_path)
        self.log = logger

    def create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                quantity DOUBLE NOT NULL,
                avg_fill_price DOUBLE NOT NULL,
                fees DOUBLE NOT NULL DEFAULT 0.0,
                exchange_order_ids VARCHAR[] NOT NULL DEFAULT [],
                entry_timestamp TIMESTAMP NOT NULL,
                exit_timestamp TIMESTAMP,
                entry_thesis VARCHAR NOT NULL DEFAULT '',
                anchor_symbol VARCHAR NOT NULL,
                correlation_score DOUBLE NOT NULL DEFAULT 0.0,
                initial_stop_loss DOUBLE NOT NULL DEFAULT 0.0,
                initial_take_profit DOUBLE NOT NULL DEFAULT 0.0,
                current_stop DOUBLE NOT NULL DEFAULT 0.0,
                current_target DOUBLE NOT NULL DEFAULT 0.0,
                highest_unrealized_profit DOUBLE NOT NULL DEFAULT 0.0,
                maximum_drawdown DOUBLE NOT NULL DEFAULT 0.0,
                review_count INTEGER NOT NULL DEFAULT 0,
                current_recommendation VARCHAR,
                lifecycle_state VARCHAR NOT NULL DEFAULT 'DISCOVERED',
                exit_reason VARCHAR
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                service VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                position_id VARCHAR,
                details JSON
            )
        """)
        self.log.info("Portfolio schema initialized")

    def save_position(self, position: Position) -> None:
        row = (
            position.position_id,
            position.symbol,
            position.side,
            position.quantity,
            position.avg_fill_price,
            position.fees,
            position.exchange_order_ids,
            position.entry_timestamp,
            position.exit_timestamp,
            position.entry_thesis,
            position.anchor_symbol,
            position.correlation_score,
            position.initial_stop_loss,
            position.initial_take_profit,
            position.current_stop,
            position.current_target,
            position.highest_unrealized_profit,
            position.maximum_drawdown,
            position.review_count,
            position.current_recommendation,
            position.lifecycle_state.value,
            position.exit_reason,
        )
        self._conn.execute(
            """
            INSERT INTO positions (
                position_id, symbol, side, quantity, avg_fill_price, fees,
                exchange_order_ids, entry_timestamp, exit_timestamp,
                entry_thesis, anchor_symbol, correlation_score,
                initial_stop_loss, initial_take_profit, current_stop,
                current_target, highest_unrealized_profit, maximum_drawdown,
                review_count, current_recommendation, lifecycle_state, exit_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (position_id) DO UPDATE SET
                quantity = EXCLUDED.quantity,
                avg_fill_price = EXCLUDED.avg_fill_price,
                fees = EXCLUDED.fees,
                exchange_order_ids = EXCLUDED.exchange_order_ids,
                exit_timestamp = EXCLUDED.exit_timestamp,
                entry_thesis = EXCLUDED.entry_thesis,
                anchor_symbol = EXCLUDED.anchor_symbol,
                correlation_score = EXCLUDED.correlation_score,
                initial_stop_loss = EXCLUDED.initial_stop_loss,
                initial_take_profit = EXCLUDED.initial_take_profit,
                current_stop = EXCLUDED.current_stop,
                current_target = EXCLUDED.current_target,
                highest_unrealized_profit = EXCLUDED.highest_unrealized_profit,
                maximum_drawdown = EXCLUDED.maximum_drawdown,
                review_count = EXCLUDED.review_count,
                current_recommendation = EXCLUDED.current_recommendation,
                lifecycle_state = EXCLUDED.lifecycle_state,
                exit_reason = EXCLUDED.exit_reason
            """,
            row,
        )

    def get_all_positions(self) -> list[Position]:
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        columns = [desc[0] for desc in self._conn.description]
        positions = []
        for row in rows:
            data = dict(zip(columns, row))
            data["lifecycle_state"] = PositionState(data["lifecycle_state"])
            data["exchange_order_ids"] = list(data["exchange_order_ids"]) if data["exchange_order_ids"] else []
            positions.append(Position(**data))
        return positions

    def append_audit_log(self, event: SystemEvent) -> None:
        max_id = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM audit_log").fetchone()[0]
        position_id = event.payload.get("position_id") if event.payload else None
        self._conn.execute(
            "INSERT INTO audit_log (id, timestamp, service, event_type, position_id, details) VALUES (?, ?, ?, ?, ?, ?)",
            [
                max_id + 1,
                event.timestamp,
                event.service_name,
                event.event_type,
                position_id,
                json.dumps(event.payload, default=str),
            ],
        )
        self.log.debug("Audit log appended", event_type=event.event_type, service=event.service_name)

    def close(self) -> None:
        self._conn.close()
