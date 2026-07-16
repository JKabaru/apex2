from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Optional

import duckdb
import structlog

from src.core.models import (
    Difference,
    EvidenceEpisode,
    InitialEvidence,
    MarketEvidence,
    Position,
    PositionState,
    ProtectionOrders,
    SystemEvent,
    TradeContext,
    VirtualFill,
)

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
    "exit_price",
    "exit_fees",
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
    "execution_mode",
    "origin",
    "execution_id",
    "trade_group_id",
    "candidate_id",
    "correlation_id",
    "llm_request_id",
    "strategy_version",
    "execution_model",
    "execution_model_version",
    "execution_parameters",
    "risk_decision",
    "risk_decision_reason",
    "created_by",
    "opportunity_source",
    "calibration_model",
    "calibration_version",
    "calibration_data",
    "virtual_fill",
    "trade_context",
    "initial_evidence",
    "current_evidence",
    "evidence_episodes",
    "protection_orders",
]


class SchemaMismatchError(Exception):
    def __init__(self, actual: int, expected: int):
        self.actual = actual
        self.expected = expected
        super().__init__(f"Schema mismatch: expected {expected} columns, found {actual}")


class PortfolioStore:
    def __init__(self, db_path: str = PORTFOLIO_DB):
        try:
            self._conn = duckdb.connect(db_path)
        except duckdb.InternalException:
            wal_path = db_path + ".wal"
            if os.path.exists(wal_path):
                os.remove(wal_path)
                self._conn = duckdb.connect(db_path)
            else:
                raise
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
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                portfolio_value DOUBLE DEFAULT 0.0,
                total_realized_pnl DOUBLE DEFAULT 0.0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                win_rate DOUBLE DEFAULT 0.0,
                avg_win DOUBLE DEFAULT 0.0,
                avg_loss DOUBLE DEFAULT 0.0,
                profit_factor DOUBLE DEFAULT 0.0,
                sharpe_ratio DOUBLE DEFAULT 0.0,
                max_drawdown DOUBLE DEFAULT 0.0,
                open_positions INTEGER DEFAULT 0,
                live_positions INTEGER DEFAULT 0,
                shadow_positions INTEGER DEFAULT 0,
                total_fees DOUBLE DEFAULT 0.0,
                extra_data JSON
            )
        """)
        self._apply_migration()
        self._verify_schema()
        self.log.info("Portfolio schema initialized")

    def _apply_migration(self) -> None:
        columns = [
            ("exit_price", "DOUBLE"),
            ("exit_fees", "DOUBLE DEFAULT 0.0"),
            ("execution_mode", "VARCHAR DEFAULT 'LIVE'"),
            ("origin", "VARCHAR DEFAULT 'NORMAL'"),
            ("execution_id", "VARCHAR"),
            ("trade_group_id", "VARCHAR"),
            ("candidate_id", "VARCHAR"),
            ("correlation_id", "VARCHAR"),
            ("llm_request_id", "VARCHAR"),
            ("strategy_version", "VARCHAR DEFAULT '1.0'"),
            ("execution_model", "VARCHAR DEFAULT 'fixed_friction_v1'"),
            ("execution_model_version", "VARCHAR DEFAULT '1.0'"),
            ("execution_parameters", "JSON DEFAULT '{}'"),
            ("risk_decision", "VARCHAR DEFAULT ''"),
            ("risk_decision_reason", "VARCHAR DEFAULT ''"),
            ("created_by", "VARCHAR DEFAULT 'SCANNER'"),
            ("opportunity_source", "VARCHAR DEFAULT 'SCANNER'"),
            ("calibration_model", "VARCHAR DEFAULT ''"),
            ("calibration_version", "VARCHAR DEFAULT ''"),
            ("calibration_data", "JSON"),
            ("virtual_fill", "JSON"),
            ("protection_orders", "JSON"),
            ("trade_context", "JSON"),
            ("initial_evidence", "JSON"),
            ("current_evidence", "JSON"),
            ("evidence_episodes", "JSON DEFAULT '[]'"),
        ]
        for col_name, col_def in columns:
            try:
                self._conn.execute(
                    f"ALTER TABLE positions ADD COLUMN IF NOT EXISTS {col_name} {col_def}"
                )
            except Exception:
                pass

        # Migrate data from legacy execution_type → execution_mode
        try:
            self._conn.execute(
                "UPDATE positions SET execution_mode = execution_type "
                "WHERE execution_mode IS NULL AND execution_type IS NOT NULL"
            )
        except Exception:
            pass
        try:
            self._conn.execute("ALTER TABLE positions DROP COLUMN IF EXISTS execution_type")
        except Exception:
            pass

    def _verify_schema(self) -> None:
        result = self._conn.execute("DESCRIBE positions").fetchall()
        actual = len(result)
        expected = len(POSITION_COLUMNS)
        if actual != expected:
            raise SchemaMismatchError(actual, expected)

    @classmethod
    def rebuild(cls, db_path: str = PORTFOLIO_DB) -> PortfolioStore:
        backup = f"{db_path}.bak.{int(time.time())}"
        wal_path = db_path + ".wal"
        if os.path.exists(db_path):
            os.rename(db_path, backup)
        if os.path.exists(wal_path):
            os.remove(wal_path)
        logger.warning("Backed up old database", backup=backup)
        return cls(db_path)

    def _deserialize_evidence(self, data: dict) -> dict:
        def _load(model_cls, raw):
            if raw and isinstance(raw, str):
                try:
                    return model_cls(**json.loads(raw))
                except (json.JSONDecodeError, TypeError, Exception):
                    return None
            return raw if isinstance(raw, model_cls) else None

        def _load_episodes(raw):
            if raw and isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    return [EvidenceEpisode(**ep) for ep in parsed]
                except (json.JSONDecodeError, TypeError, Exception):
                    return []
            return raw if isinstance(raw, list) else []

        parsed = dict(data)
        parsed["trade_context"] = _load(TradeContext, data.get("trade_context"))
        parsed["initial_evidence"] = _load(InitialEvidence, data.get("initial_evidence"))
        parsed["current_evidence"] = _load(MarketEvidence, data.get("current_evidence"))
        parsed["evidence_episodes"] = _load_episodes(data.get("evidence_episodes"))
        parsed["virtual_fill"] = _load(VirtualFill, data.get("virtual_fill"))
        parsed["protection_orders"] = _load(ProtectionOrders, data.get("protection_orders"))
        if parsed["protection_orders"] is None:
            parsed["protection_orders"] = ProtectionOrders()
        return parsed

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
            position.exit_price,
            position.exit_fees,
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
            position.execution_mode,
            position.origin,
            position.execution_id,
            position.trade_group_id,
            position.candidate_id,
            position.correlation_id,
            position.llm_request_id,
            position.strategy_version,
            position.execution_model,
            position.execution_model_version,
            json.dumps(position.execution_parameters) if position.execution_parameters else "{}",
            position.risk_decision,
            position.risk_decision_reason,
            position.created_by,
            position.opportunity_source,
            position.calibration_model,
            position.calibration_version,
            json.dumps(position.calibration_data) if position.calibration_data else None,
            json.dumps(position.virtual_fill.model_dump(mode="json")) if position.virtual_fill else None,
            json.dumps(position.protection_orders.model_dump(mode="json")) if position.protection_orders else None,
            json.dumps(position.trade_context.model_dump(mode="json")) if position.trade_context else None,
            json.dumps(position.initial_evidence.model_dump(mode="json")) if position.initial_evidence else None,
            json.dumps(position.current_evidence.model_dump(mode="json")) if position.current_evidence else None,
            json.dumps([ep.model_dump(mode="json") for ep in position.evidence_episodes]) if position.evidence_episodes else "[]",
        )
        self._conn.execute(
            """
            INSERT INTO positions (
                position_id, symbol, side, quantity, avg_fill_price, fees,
                exchange_order_ids, entry_timestamp, exit_timestamp,
                exit_price, exit_fees,
                entry_thesis, anchor_symbol, correlation_score,
                initial_stop_loss, initial_take_profit, current_stop,
                current_target, highest_unrealized_profit, maximum_drawdown,
                review_count, current_recommendation, lifecycle_state, exit_reason,
                execution_mode, origin, execution_id, trade_group_id,
                candidate_id, correlation_id, llm_request_id, strategy_version,
                execution_model, execution_model_version, execution_parameters,
                risk_decision, risk_decision_reason, created_by, opportunity_source,
                calibration_model, calibration_version, calibration_data,
                virtual_fill,
                protection_orders,
                trade_context, initial_evidence, current_evidence, evidence_episodes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (position_id) DO UPDATE SET
                quantity = EXCLUDED.quantity,
                avg_fill_price = EXCLUDED.avg_fill_price,
                fees = EXCLUDED.fees,
                exchange_order_ids = EXCLUDED.exchange_order_ids,
                exit_timestamp = EXCLUDED.exit_timestamp,
                exit_price = EXCLUDED.exit_price,
                exit_fees = EXCLUDED.exit_fees,
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
                exit_reason = EXCLUDED.exit_reason,
                execution_mode = EXCLUDED.execution_mode,
                origin = EXCLUDED.origin,
                execution_id = EXCLUDED.execution_id,
                trade_group_id = EXCLUDED.trade_group_id,
                candidate_id = EXCLUDED.candidate_id,
                correlation_id = EXCLUDED.correlation_id,
                llm_request_id = EXCLUDED.llm_request_id,
                strategy_version = EXCLUDED.strategy_version,
                execution_model = EXCLUDED.execution_model,
                execution_model_version = EXCLUDED.execution_model_version,
                execution_parameters = EXCLUDED.execution_parameters,
                risk_decision = EXCLUDED.risk_decision,
                risk_decision_reason = EXCLUDED.risk_decision_reason,
                created_by = EXCLUDED.created_by,
                opportunity_source = EXCLUDED.opportunity_source,
                calibration_model = EXCLUDED.calibration_model,
                calibration_version = EXCLUDED.calibration_version,
                calibration_data = EXCLUDED.calibration_data,
                virtual_fill = EXCLUDED.virtual_fill,
                trade_context = EXCLUDED.trade_context,
                initial_evidence = EXCLUDED.initial_evidence,
                current_evidence = EXCLUDED.current_evidence,
                evidence_episodes = EXCLUDED.evidence_episodes,
                protection_orders = EXCLUDED.protection_orders
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
            data["exchange_order_ids"] = list(data["exchange_order_ids"]) if data.get("exchange_order_ids") else []
            if data.get("execution_parameters") and isinstance(data["execution_parameters"], str):
                try:
                    data["execution_parameters"] = json.loads(data["execution_parameters"])
                except (json.JSONDecodeError, TypeError):
                    data["execution_parameters"] = {}
            if data.get("calibration_data") and isinstance(data["calibration_data"], str):
                try:
                    data["calibration_data"] = json.loads(data["calibration_data"])
                except (json.JSONDecodeError, TypeError):
                    data["calibration_data"] = None
            data = self._deserialize_evidence(data)
            positions.append(Position(**data))
        return positions

    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        row = self._conn.execute(
            "SELECT * FROM positions WHERE position_id = ?", [position_id]
        ).fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in self._conn.description]
        data = dict(zip(columns, row))
        data["lifecycle_state"] = PositionState(data["lifecycle_state"])
        data["exchange_order_ids"] = list(data["exchange_order_ids"]) if data.get("exchange_order_ids") else []
        if data.get("execution_parameters") and isinstance(data["execution_parameters"], str):
            try:
                data["execution_parameters"] = json.loads(data["execution_parameters"])
            except (json.JSONDecodeError, TypeError):
                data["execution_parameters"] = {}
        if data.get("calibration_data") and isinstance(data["calibration_data"], str):
            try:
                data["calibration_data"] = json.loads(data["calibration_data"])
            except (json.JSONDecodeError, TypeError):
                data["calibration_data"] = None
        data = self._deserialize_evidence(data)
        return Position(**data)

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

    def reset_all_local_positions(self) -> int:
        count = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE lifecycle_state NOT IN ('CLOSED', 'ARCHIVED')"
        ).fetchone()[0]
        self._conn.execute("""
            UPDATE positions SET
                lifecycle_state = 'CLOSED',
                exit_reason = 'MANUAL_RESET',
                exit_timestamp = CURRENT_TIMESTAMP
            WHERE lifecycle_state NOT IN ('CLOSED', 'ARCHIVED')
        """)
        self.log.warning("Local positions reset to CLOSED", count=count)
        return count

    # ── Portfolio Metrics ──────────────────────────────────────────────

    def get_completed_positions(self) -> list[Position]:
        """Return all closed/archived positions for PnL analysis."""
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE lifecycle_state IN ('CLOSED', 'ARCHIVED')"
        ).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        positions = []
        for row in rows:
            data = dict(zip(columns, row))
            data["lifecycle_state"] = PositionState(data["lifecycle_state"])
            data["exchange_order_ids"] = list(data["exchange_order_ids"]) if data.get("exchange_order_ids") else []
            if data.get("execution_parameters") and isinstance(data["execution_parameters"], str):
                try:
                    data["execution_parameters"] = json.loads(data["execution_parameters"])
                except (json.JSONDecodeError, TypeError):
                    data["execution_parameters"] = {}
            if data.get("calibration_data") and isinstance(data["calibration_data"], str):
                try:
                    data["calibration_data"] = json.loads(data["calibration_data"])
                except (json.JSONDecodeError, TypeError):
                    data["calibration_data"] = None
            data = self._deserialize_evidence(data)
            positions.append(Position(**data))
        return positions

    def get_open_position_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE lifecycle_state NOT IN ('CLOSED', 'ARCHIVED')"
        ).fetchone()
        return row[0] if row else 0

    def get_open_positions_by_mode(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT execution_mode, COUNT(*) as cnt FROM positions "
            "WHERE lifecycle_state NOT IN ('CLOSED', 'ARCHIVED') "
            "GROUP BY execution_mode"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def save_metrics_snapshot(self, metrics: dict) -> int:
        max_id = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM portfolio_metrics").fetchone()[0]
        new_id = max_id + 1
        extra = {k: v for k, v in metrics.items() if k not in (
            "timestamp", "portfolio_value", "total_realized_pnl", "win_count", "loss_count",
            "win_rate", "avg_win", "avg_loss", "profit_factor", "sharpe_ratio",
            "max_drawdown", "open_positions", "live_positions", "shadow_positions", "total_fees",
        )}
        self._conn.execute(
            """
            INSERT INTO portfolio_metrics (
                id, timestamp, portfolio_value, total_realized_pnl,
                win_count, loss_count, win_rate, avg_win, avg_loss,
                profit_factor, sharpe_ratio, max_drawdown,
                open_positions, live_positions, shadow_positions, total_fees,
                extra_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                new_id,
                metrics.get("timestamp", datetime.utcnow().isoformat()),
                metrics.get("portfolio_value", 0.0),
                metrics.get("total_realized_pnl", 0.0),
                metrics.get("win_count", 0),
                metrics.get("loss_count", 0),
                metrics.get("win_rate", 0.0),
                metrics.get("avg_win", 0.0),
                metrics.get("avg_loss", 0.0),
                metrics.get("profit_factor", 0.0),
                metrics.get("sharpe_ratio", 0.0),
                metrics.get("max_drawdown", 0.0),
                metrics.get("open_positions", 0),
                metrics.get("live_positions", 0),
                metrics.get("shadow_positions", 0),
                metrics.get("total_fees", 0.0),
                json.dumps(extra) if extra else None,
            ],
        )
        return new_id

    def get_metrics_history(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM portfolio_metrics ORDER BY id DESC LIMIT ?", [limit]
        ).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        self._conn.close()
