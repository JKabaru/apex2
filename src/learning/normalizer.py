from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import structlog

from src.agent.indicators import compute_atr, compute_rsi
from src.models.learning.trade_experience import (
    LearningExperience,
    NormalizedMetrics,
)

logger = structlog.get_logger("experience_normalizer")

OHLCV_DB = "data/ohlcv.duckdb"

TIMEFRAME_TO_MINUTES = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360,
    "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
}


def _parse_timeframe_minutes(tf: str) -> Optional[int]:
    tf = tf.strip().lower()
    if tf in TIMEFRAME_TO_MINUTES:
        return TIMEFRAME_TO_MINUTES[tf]
    if tf.endswith("m"):
        try:
            return int(tf[:-1])
        except ValueError:
            return None
    if tf.endswith("h"):
        try:
            return int(tf[:-1]) * 60
        except ValueError:
            return None
    if tf.endswith("d"):
        try:
            return int(tf[:-1]) * 1440
        except ValueError:
            return None
    if tf.endswith("w"):
        try:
            return int(tf[:-1]) * 10080
        except ValueError:
            return None
    return None


def _compute_full_atr_series(df: pd.DataFrame, period: int = 14) -> list[float]:
    if df.empty or len(df) < period + 1:
        return []
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]
    atr_series = []
    atr = np.mean(tr[1:period + 1])
    atr_series.append(float(atr))
    for i in range(period + 1, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
        atr_series.append(float(atr))
    return atr_series


def _compute_full_rsi_series(df: pd.DataFrame, period: int = 14) -> list[float]:
    if df.empty or len(df) < period + 1:
        return []
    closes = df["close"].values
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi_series: list[float] = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_series.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_series.append(100.0 - 100.0 / (1.0 + rs))
    return rsi_series


def _calculate_percentile(value: float, series: list[float]) -> Optional[float]:
    if not series or len(series) < 2:
        return None
    sorted_series = sorted(series)
    n = len(sorted_series)
    count_less = sum(1 for v in sorted_series if v < value)
    count_equal = sum(1 for v in sorted_series if v == value)
    rank = count_less + 0.5 * count_equal
    return rank / n


def _calculate_atr_multiple(price_delta: float, atr_value: float) -> Optional[float]:
    if atr_value is None or atr_value <= 0 or math.isnan(atr_value):
        return None
    if math.isnan(price_delta) or math.isinf(price_delta):
        return None
    return price_delta / atr_value


class ExperienceNormalizer:
    """Stage 3 of the learning pipeline.
    Pure mathematical normalization. No judgment. No LLM.
    Opens a read-only connection to OHLCV DuckDB for historical series."""

    normalizer_version: str = "1.0"

    def __init__(self, ohlcv_db_path: str = OHLCV_DB):
        self._conn = duckdb.connect(ohlcv_db_path)
        logger.info("ExperienceNormalizer initialized", ohlcv_db=ohlcv_db_path)

    def normalize(self, experience: LearningExperience) -> NormalizedMetrics:
        symbol = experience.symbol
        tf_minutes = _parse_timeframe_minutes(experience.timeframe)
        candles = self._fetch_candle_series(symbol, experience.timeframe)

        atr_series: list[float] = []
        rsi_series: list[float] = []
        if candles is not None and len(candles) > 14:
            atr_series = _compute_full_atr_series(candles)
            rsi_series = _compute_full_rsi_series(candles)

        atr_at_entry = atr_series[-1] if atr_series else None

        mkt = self._market_normalization(experience, atr_at_entry, atr_series, rsi_series)
        exec_norm = self._execution_normalization(experience)

        return NormalizedMetrics(
            normalizer_version=self.normalizer_version,
            **mkt,
            **exec_norm,
        )

    def _fetch_candle_series(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> Optional[pd.DataFrame]:
        try:
            if timeframe == "1m":
                rows = self._conn.execute(
                    "SELECT open_time, open, high, low, close, volume "
                    "FROM ohlcv_1m WHERE symbol = ? ORDER BY open_time DESC LIMIT ?",
                    [symbol, limit],
                ).fetchdf()
            else:
                tf_minutes = _parse_timeframe_minutes(timeframe)
                if tf_minutes is None:
                    return None
                rows = self._conn.execute(
                    "SELECT bucket AS open_time, open, high, low, close, volume "
                    "FROM ohlcv_agg WHERE symbol = ? AND tf_minutes = ? "
                    "ORDER BY bucket DESC LIMIT ?",
                    [symbol, tf_minutes, limit],
                ).fetchdf()
            if rows.empty:
                return None
            return rows.sort_values("open_time").reset_index(drop=True)
        except Exception:
            logger.warning("Failed to fetch candles for normalization",
                           symbol=symbol, timeframe=timeframe)
            return None

    def _market_normalization(
        self,
        experience: LearningExperience,
        atr_at_entry: Optional[float],
        atr_series: list[float],
        rsi_series: list[float],
    ) -> dict:
        result: dict = {}

        direction = 1.0  # placeholder — side determines direction
        notional_qty = 1.0  # placeholder — quantity comes from Position join

        # entry ATR multiple
        if atr_at_entry:
            result["normalized_entry_atr_multiple"] = _calculate_atr_multiple(
                experience.entry_price, atr_at_entry
            )
        else:
            result["normalized_entry_atr_multiple"] = None

        # exit ATR multiple (pnl delta in price space)
        if experience.exit_price is not None and experience.entry_price is not None and atr_at_entry:
            pnl_delta = (experience.exit_price - experience.entry_price) * direction
            result["normalized_exit_atr_multiple"] = _calculate_atr_multiple(
                pnl_delta, atr_at_entry
            )
        else:
            result["normalized_exit_atr_multiple"] = None

        # pnl ATR multiple
        if experience.exit_price is not None and experience.entry_price is not None and atr_at_entry:
            pnl_usdt = (experience.exit_price - experience.entry_price) * direction * notional_qty
            result["pnl_atr_multiple"] = _calculate_atr_multiple(
                pnl_usdt, atr_at_entry * notional_qty
            )
        else:
            result["pnl_atr_multiple"] = None

        # MFE / MAE ATR multiples
        if atr_at_entry:
            result["mfe_atr_multiple"] = _calculate_atr_multiple(
                experience.highest_unrealized_profit, atr_at_entry * notional_qty
            )
            result["mae_atr_multiple"] = _calculate_atr_multiple(
                experience.maximum_drawdown, atr_at_entry * notional_qty
            )
        else:
            result["mfe_atr_multiple"] = None
            result["mae_atr_multiple"] = None

        # RSI percentile
        if experience.entry_rsi is not None and rsi_series:
            result["entry_rsi_percentile"] = _calculate_percentile(
                experience.entry_rsi, rsi_series
            )
        else:
            result["entry_rsi_percentile"] = None

        # volatility percentile
        if experience.entry_atr is not None and atr_series:
            result["entry_volatility_percentile"] = _calculate_percentile(
                experience.entry_atr, atr_series
            )
        else:
            result["entry_volatility_percentile"] = None

        # holding duration (computed from timestamps)
        result["holding_duration_minutes"] = None
        result["bars_held"] = None

        return result

    def _execution_normalization(
        self,
        experience: LearningExperience,
    ) -> dict:
        result: dict = {}

        total_slippage = 0.0
        if experience.slippage_bps is not None:
            total_slippage += experience.slippage_bps
        if experience.spread_bps is not None:
            total_slippage += experience.spread_bps
        has_slippage = experience.slippage_bps is not None or experience.spread_bps is not None
        result["total_slippage_bps"] = total_slippage if has_slippage else None

        notional = (experience.entry_price or 0.0) * 1.0
        total_fees = (experience.fees or 0.0) + (experience.exit_fees or 0.0)
        if notional > 0 and experience.entry_price:
            result["total_fees_bps"] = round(total_fees / notional * 10000, 4)
        else:
            result["total_fees_bps"] = None

        # realized_rr and initial_risk_atr_multiple require stop/tp data
        result["realized_rr"] = None
        result["initial_risk_atr_multiple"] = None

        return result

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
