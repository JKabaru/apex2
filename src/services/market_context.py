from __future__ import annotations

import structlog

import duckdb
import pandas as pd

from src.agent.indicators import compute_rsi, compute_macd, compute_bollinger_bands, compute_atr

logger = structlog.get_logger("market_context")

OHLCV_DB = "data/ohlcv.duckdb"
CORR_DB = "data/correlation_matrix.duckdb"


class MarketContextService:
    def __init__(self):
        self._ohlcv_conn = duckdb.connect(OHLCV_DB)
        self._corr_conn = duckdb.connect(CORR_DB)
        logger.info("MarketContextService initialized")

    def _fetch_candles(self, symbol: str, timeframe: str, limit: int = 50) -> pd.DataFrame:
        try:
            if timeframe == "1m":
                rows = self._ohlcv_conn.execute(
                    "SELECT open_time, open, high, low, close, volume "
                    "FROM ohlcv_1m WHERE symbol = ? ORDER BY open_time DESC LIMIT ?",
                    [symbol, limit],
                ).fetchdf()
            else:
                tf_minutes = int(timeframe.replace("m", ""))
                rows = self._ohlcv_conn.execute(
                    "SELECT bucket AS open_time, open, high, low, close, volume "
                    "FROM ohlcv_agg WHERE symbol = ? AND tf_minutes = ? ORDER BY bucket DESC LIMIT ?",
                    [symbol, tf_minutes, limit],
                ).fetchdf()
            if rows.empty:
                logger.warning("No candle data found", symbol=symbol, timeframe=timeframe)
                return pd.DataFrame()
            return rows.sort_values("open_time").reset_index(drop=True)
        except Exception as e:
            logger.error("Failed to fetch candles", symbol=symbol, timeframe=timeframe, error=str(e))
            return pd.DataFrame()

    def _fetch_correlations(self, symbol: str) -> list[dict]:
        try:
            rows = self._corr_conn.execute(
                "SELECT anchor, alt, timeframe, coefficient, dominant_lag, direction, p_value, significant "
                "FROM correlation_snapshot "
                "WHERE alt = ? AND significant = 1 "
                "ORDER BY ABS(coefficient) DESC LIMIT 3",
                [symbol],
            ).fetchall()
            columns = ["anchor", "alt", "timeframe", "coefficient", "dominant_lag", "direction", "p_value", "significant"]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error("Failed to fetch correlations", symbol=symbol, error=str(e))
            return []

    async def get_state(self, symbol: str, timeframe: str = "5m") -> dict:
        df = self._fetch_candles(symbol, timeframe)
        if df.empty:
            return {"current_price": 0.0, "indicators": {}, "correlations": []}

        current_price = float(df["close"].iloc[-1])

        indicators = {}
        try:
            indicators.update(compute_rsi(df))
        except Exception as e:
            logger.error("RSI computation failed", symbol=symbol, error=str(e))
        try:
            indicators.update(compute_macd(df))
        except Exception as e:
            logger.error("MACD computation failed", symbol=symbol, error=str(e))
        try:
            indicators.update(compute_bollinger_bands(df))
        except Exception as e:
            logger.error("BB computation failed", symbol=symbol, error=str(e))
        try:
            indicators.update(compute_atr(df))
        except Exception as e:
            logger.error("ATR computation failed", symbol=symbol, error=str(e))

        correlations = self._fetch_correlations(symbol)

        return {
            "current_price": current_price,
            "indicators": indicators,
            "correlations": correlations,
        }

    def close(self) -> None:
        try:
            self._ohlcv_conn.close()
            self._corr_conn.close()
        except Exception as e:
            logger.error("Error closing MarketContextService connections", error=str(e))
