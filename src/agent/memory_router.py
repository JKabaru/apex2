from datetime import datetime, timezone

import duckdb
import structlog

MEMORY_DB = "data/agent_memory.duckdb"

logger = structlog.get_logger("memory_router")


def _get_conn():
    return duckdb.connect(MEMORY_DB)


def _init_db():
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY,
                tags TEXT,
                content TEXT,
                outcome TEXT
            )
        """)
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS knowledge_base_id_seq
        """)
        conn.commit()
    except Exception as e:
        logger.error("Failed to initialize memory DB", error=str(e))
    finally:
        conn.close()


_init_db()


async def retrieve_memories(current_state: dict, limit: int = 3) -> list[dict]:
    keywords = _extract_keywords(current_state)
    logger.debug("Memory keywords extracted", keywords=keywords)
    if not keywords:
        return []

    conn = _get_conn()
    try:
        for kw in keywords:
            pattern = f"%{kw}%"
            logger.debug("Memory ILIKE query", keyword=kw, pattern=pattern)
            rows = conn.execute(
                "SELECT id, tags, content, outcome FROM knowledge_base "
                "WHERE tags ILIKE ? OR content ILIKE ? "
                "ORDER BY id DESC LIMIT ?",
                [pattern, pattern, limit],
            ).fetchall()
            logger.debug("Memory raw results", keyword=kw, rows=len(rows), results=rows)
            if rows:
                return [
                    {"id": r[0], "tags": r[1], "content": r[2], "outcome": r[3]}
                    for r in rows
                ]
        return [
            {"id": -1, "tags": "fallback", "content": "No specific historical memories found for this exact regime. Rely strictly on current indicator and correlation confluence.", "outcome": "fallback"}
        ]
    except Exception as e:
        logger.error("Memory query failed", keywords=keywords, error=str(e))
        return []
    finally:
        conn.close()


async def store_memory(tags: str, content: str, outcome: str):
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO knowledge_base (tags, content, outcome) VALUES (?, ?, ?)",
            [tags, content, outcome],
        )
        conn.commit()
        logger.info("Memory stored", tags=tags)
    except Exception as e:
        conn.rollback()
        logger.error("Failed to store memory", error=str(e))
    finally:
        conn.close()


def _extract_keywords(state: dict) -> list[str]:
    keywords = []
    symbol = state.get("symbol", "")
    if symbol:
        keywords.append(symbol)

    indicators = state.get("indicators", {})
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi > 70:
            keywords.append("RSI_OVERBOUGHT")
        elif rsi < 30:
            keywords.append("RSI_OVERSOLD")

    corrs = state.get("top_correlations", [])
    for c in corrs:
        direction = c.get("direction", 0)
        if direction > 0:
            keywords.append("positive_correlation")
        elif direction < 0:
            keywords.append("negative_correlation")

    logger.debug("Extracted keywords", symbol=symbol, rsi=rsi, correlation_count=len(corrs), keywords=keywords)
    return keywords
