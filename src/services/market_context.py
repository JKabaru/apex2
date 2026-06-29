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

    def _derive_trend_regime(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 200:
            return "UNKNOWN"
        close = float(df["close"].iloc[-1])
        sma200 = float(df["close"].rolling(200).mean().iloc[-1])
        if close > sma200 * 1.01:
            return "BULLISH"
        if close < sma200 * 0.99:
            return "BEARISH"
        return "RANGING"

    def _derive_momentum(self, indicators: dict) -> str:
        hist = indicators.get("histogram")
        if hist is None:
            return "UNKNOWN"
        if hist > 0:
            return "POSITIVE"
        if hist < 0:
            return "NEGATIVE"
        return "NEUTRAL"

    def _derive_volatility_regime(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 20:
            return "UNKNOWN"
        current_result = compute_atr(df)
        current_atr = current_result.get("atr")
        if current_atr is None:
            return "UNKNOWN"
        prior_df = df.iloc[:-5]
        if len(prior_df) < 14:
            return "UNKNOWN"
        prior_result = compute_atr(prior_df)
        prior_atr = prior_result.get("atr")
        if prior_atr is None or prior_atr == 0:
            return "UNKNOWN"
        ratio = current_atr / prior_atr
        if ratio > 1.05:
            return "EXPANDING"
        if ratio < 0.95:
            return "CONTRACTING"
        return "STABLE"

    def _derive_volume_profile(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 20:
            return "UNKNOWN"
        vol_current = float(df["volume"].iloc[-1])
        vol_avg = float(df["volume"].tail(20).mean())
        if vol_avg <= 0:
            return "UNKNOWN"
        ratio = vol_current / vol_avg
        if ratio > 1.5:
            return "HIGH"
        if ratio < 0.5:
            return "LOW"
        return "NORMAL"

    def _derive_correlation_regime(self, coefficient: float) -> str:
        ac = abs(coefficient)
        if ac >= 0.7:
            return "STRONG"
        if ac >= 0.4:
            return "MODERATE"
        if ac >= 0.2:
            return "WEAK"
        return "NEGLIGIBLE"

    async def get_state(self, symbol: str, timeframe: str = "5m") -> dict:
        trend_df = self._fetch_candles(symbol, timeframe, limit=200)
        if trend_df.empty:
            return {"current_price": 0.0, "indicators": {}, "correlations": []}

        df = trend_df.tail(50).reset_index(drop=True) if len(trend_df) > 50 else trend_df
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
        avg_corr = 0.0
        if correlations:
            avg_corr = sum(abs(c.get("coefficient", 0)) for c in correlations) / len(correlations)
        top_corr = correlations[0].get("coefficient", 0.0) if correlations else 0.0

        trend_regime = self._derive_trend_regime(trend_df)
        momentum = self._derive_momentum(indicators)
        volatility_regime = self._derive_volatility_regime(df)
        volume_profile = self._derive_volume_profile(df)
        correlation_regime = self._derive_correlation_regime(top_corr)

        return {
            "current_price": current_price,
            "indicators": indicators,
            "correlations": correlations,
            "trend_regime": trend_regime,
            "momentum": momentum,
            "volatility_regime": volatility_regime,
            "volume_profile": volume_profile,
            "correlation_regime": correlation_regime,
            "correlation_score": avg_corr,
        }

    def close(self) -> None:
        try:
            self._ohlcv_conn.close()
            self._corr_conn.close()
        except Exception as e:
            logger.error("Error closing MarketContextService connections", error=str(e))
