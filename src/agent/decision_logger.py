import json
from datetime import datetime, timezone

import duckdb
import structlog

from .llm_executor import AgentDecision

DECISIONS_DB = "data/agent_decisions.duckdb"

logger = structlog.get_logger("decision_logger")


def _get_conn():
    conn = duckdb.connect(DECISIONS_DB)
    return conn


def _init_db():
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                rationale TEXT,
                suggested_timeframe TEXT,
                state_snapshot TEXT,
                status TEXT DEFAULT 'PENDING',
                actual_pnl REAL
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error("Failed to initialize decision_logger DB", error=str(e))
        raise
    finally:
        conn.close()


_init_db()


async def log_decision(decision: AgentDecision, state: dict):
    conn = _get_conn()
    try:
        state_json = json.dumps(state, default=str)
        row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM agent_decisions").fetchone()
        next_id = row[0]
        conn.execute("""
            INSERT INTO agent_decisions
                (id, timestamp, symbol, timeframe, action, confidence,
                 rationale, suggested_timeframe, state_snapshot, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
        """, [
            next_id,
            datetime.now(timezone.utc).isoformat(),
            state.get("symbol", ""),
            state.get("timeframe", ""),
            decision.action,
            decision.confidence,
            decision.rationale,
            decision.suggested_timeframe,
            state_json,
        ])
        conn.commit()

        logger.info("Decision logged", id=next_id, action=decision.action, symbol=state.get("symbol"))
        return next_id
    except Exception as e:
        logger.error("Failed to log decision", error=str(e))
        raise
    finally:
        conn.close()


async def update_outcome(decision_id: int, outcome: str, actual_pnl: float):
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE agent_decisions SET status = ?, actual_pnl = ? WHERE id = ?",
            [outcome, actual_pnl, decision_id],
        )
        conn.commit()
        logger.info("Decision outcome updated", id=decision_id, outcome=outcome, pnl=actual_pnl)
    except Exception as e:
        logger.error("Failed to update decision outcome", id=decision_id, error=str(e))
    finally:
        conn.close()
