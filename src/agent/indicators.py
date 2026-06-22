import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger("indicators")


def compute_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    logger.debug("Computing RSI", df_shape=df.shape, period=period)
    if df.empty or len(df) < period + 1:
        logger.debug("RSI insufficient data", df_shape=df.shape, required=period + 1)
        return {"rsi": None}
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    latest = rsi.iloc[-1]
    result = {"rsi": round(float(latest), 2) if pd.notna(latest) else None}
    logger.debug("RSI result", rsi=result["rsi"])
    return result


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    logger.debug("Computing MACD", df_shape=df.shape, fast=fast, slow=slow, signal=signal)
    if df.empty or len(df) < slow:
        logger.debug("MACD insufficient data", df_shape=df.shape, required=slow)
        return {"macd": None, "signal": None, "histogram": None}
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    result = {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }
    logger.debug("MACD result", macd=result["macd"], signal=result["signal"], histogram=result["histogram"])
    return result


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> dict:
    logger.debug("Computing Bollinger Bands", df_shape=df.shape, period=period, std_dev=std_dev)
    if df.empty or len(df) < period:
        logger.debug("BB insufficient data", df_shape=df.shape, required=period)
        return {"upper": None, "middle": None, "lower": None}
    middle = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    result = {
        "upper": round(float(upper.iloc[-1]), 8) if pd.notna(upper.iloc[-1]) else None,
        "middle": round(float(middle.iloc[-1]), 8) if pd.notna(middle.iloc[-1]) else None,
        "lower": round(float(lower.iloc[-1]), 8) if pd.notna(lower.iloc[-1]) else None,
    }
    logger.debug("BB result", upper=result["upper"], middle=result["middle"], lower=result["lower"])
    return result


def compute_atr(df: pd.DataFrame, period: int = 14) -> dict:
    logger.debug("Computing ATR", df_shape=df.shape, period=period)
    if df.empty or len(df) < period + 1:
        logger.debug("ATR insufficient data", df_shape=df.shape, required=period + 1)
        return {"atr": None}
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    latest = atr.iloc[-1]
    result = {"atr": round(float(latest), 8) if pd.notna(latest) else None}
    logger.debug("ATR result", atr=result["atr"])
    return result
