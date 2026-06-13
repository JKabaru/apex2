from datetime import datetime, timezone

import duckdb
import pandas as pd
import structlog

from . import indicators

OHLCV_DB = "data/ohlcv.duckdb"
CORRELATION_DB = "data/correlation_matrix.duckdb"

logger = structlog.get_logger("state_builder")

_ohlcv_conn = duckdb.connect(OHLCV_DB)
_corr_conn = duckdb.connect(CORRELATION_DB)


async def build_state(symbol: str, timeframe: str) -> dict:
    tf_minutes = _parse_timeframe(timeframe)
    logger.info("Fetching candles from DuckDB", symbol=symbol, timeframe=timeframe, tf_minutes=tf_minutes)

    if tf_minutes == 1:
        rows = _ohlcv_conn.execute(
            "SELECT open_time, open, high, low, close, volume "
            "FROM ohlcv_1m WHERE symbol = ? ORDER BY open_time DESC LIMIT 50",
            [symbol],
        ).fetchall()
    else:
        rows = _ohlcv_conn.execute(
            "SELECT bucket, open, high, low, close, volume "
            "FROM ohlcv_agg WHERE tf_minutes = ? AND symbol = ? "
            "ORDER BY bucket DESC LIMIT 50",
            [tf_minutes, symbol],
        ).fetchall()

    logger.info("Candles fetched from DuckDB", symbol=symbol, count=len(rows))
    if rows:
        logger.debug("Raw candle sample", symbol=symbol, first=rows[0], last=rows[-1])

    if not rows:
        logger.warning("No candle data found for state", symbol=symbol, timeframe=timeframe)
        current_price = None
        df = pd.DataFrame()
    else:
        rows.reverse()
        if tf_minutes == 1:
            df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        else:
            df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close", "volume"])
        current_price = float(df["close"].iloc[-1])
        logger.debug("DataFrame after conversion", symbol=symbol, rows=len(df), columns=list(df.columns), dtypes=df.dtypes.to_dict())

    indicator_values = {}
    if not df.empty:
        for k, v in indicators.compute_rsi(df).items():
            indicator_values[k] = v
        for k, v in indicators.compute_macd(df).items():
            indicator_values[k] = v
        for k, v in indicators.compute_bollinger_bands(df).items():
            indicator_values[k] = v
        for k, v in indicators.compute_atr(df).items():
            indicator_values[k] = v

    logger.debug("Raw indicator result", symbol=symbol, rsi=indicator_values.get("rsi"), macd=indicator_values.get("macd"), bb_upper=indicator_values.get("upper"), atr=indicator_values.get("atr"))

    if not indicator_values:
        logger.warning("State builder returned empty indicators", symbol=symbol, rows=len(rows))
    else:
        logger.info("Computed indicators", symbol=symbol, indicators=indicator_values)

    top_correlations = await _fetch_top_correlations(symbol)
    if not top_correlations:
        logger.warning("State builder returned empty correlations", symbol=symbol)
    else:
        logger.info("Fetched correlations", symbol=symbol, count=len(top_correlations))

    current_price_f = float(current_price) if current_price is not None else 0.0

    indicator_summary = ""
    if indicator_values:
        rsi_val = indicator_values.get("rsi")
        macd_val = indicator_values.get("macd")
        signal_val = indicator_values.get("signal")
        hist_val = indicator_values.get("histogram")
        upper = indicator_values.get("upper")
        mid = indicator_values.get("middle")
        lower = indicator_values.get("lower")
        atr_val = indicator_values.get("atr")

        if rsi_val is None:
            indicator_summary = "Insufficient candle data to compute technical indicators."
        else:
            bb_pos = "Upper"
            if mid is not None and current_price_f < mid:
                bb_pos = "Lower"
            bb_dist = ((current_price_f - mid) / mid * 100) if mid else 0

            macd_label = "positive histogram = bullish momentum building" if hist_val and hist_val > 0 else "negative histogram = bearish momentum building"

            if atr_val and current_price_f:
                vol_label = f"ATR={atr_val:.2f} USDT ({(atr_val / current_price_f * 100):.2f}% of price)"
            elif atr_val:
                vol_label = f"ATR={atr_val:.2f}"
            else:
                vol_label = "ATR=N/A"

            rsi_label = ""
            if rsi_val > 70:
                rsi_label = "overbought territory (bearish signal, potential reversal down)"
            elif rsi_val < 30:
                rsi_label = "oversold territory (bullish signal, potential reversal up)"
            elif rsi_val > 50:
                rsi_label = "bullish-leaning neutral territory (above 50 midpoint)"
            else:
                rsi_label = "bearish-leaning neutral territory (below 50 midpoint)"

            atr_pct = 0.0
            vol_regime = "N/A"
            if atr_val and current_price_f:
                atr_pct = atr_val / current_price_f * 100
                vol_regime = 'HIGH' if atr_pct > 1 else 'MODERATE' if atr_pct > 0.5 else 'LOW'

            indicator_summary = (
                f"--- RSI(14) ---\n"
                f"Value: {rsi_val:.2f} — RSI is in {rsi_label}.\n"
                f"\n"
                f"--- MACD(12,26,9) ---\n"
                f"MACD Line: {macd_val:.2f}, Signal Line: {signal_val:.2f}, Histogram: {hist_val:.2f}\n"
                f"Interpretation: MACD {'above' if macd_val and signal_val and macd_val > signal_val else 'below'} signal line, {macd_label}.\n"
                f"\n"
                f"--- Bollinger Bands(20,2) ---\n"
                f"Upper: {upper:.2f}, Middle (SMA): {mid:.2f}, Lower: {lower:.2f}\n"
                f"Current Price ({current_price_f:.2f}) is near the {bb_pos} Band ({bb_dist:+.2f}% from middle).\n"
                f"Band Width: {upper - lower:.2f} ({(upper - lower) / mid * 100:.2f}% of mid).\n"
                f"\n"
                f"--- ATR(14) ---\n"
                f"{vol_label}\n"
                f"Volatility regime: {vol_regime} ({atr_pct:.2f}% of price)."
            )

    correlation_summary = ""
    if top_correlations:
        items = []
        for i, c in enumerate(top_correlations, 1):
            direction_label = "POSITIVE" if c.get("direction", 0) > 0 else "NEGATIVE"
            coeff = c.get("coefficient", 0)
            p_val = c.get("p_value", 1)
            sig_label = "SIGNIFICANT" if c.get("significant", False) and p_val < 0.05 else "NOT SIGNIFICANT"
            lag = c.get("dominant_lag", "?")
            strength_label = "strong" if abs(coeff) > 0.7 else "moderate" if abs(coeff) > 0.4 else "weak"
            items.append(
                f"Correlation #{i}: {c.get('pair', '?')}\n"
                f"  Lead Relationship: {c.get('anchor', '?')} leads {c.get('alt', '?')} by {lag} candles\n"
                f"  Direction: {direction_label} ({strength_label}, coefficient={coeff:.4f})\n"
                f"  Statistical Significance: p-value={p_val:.6f} — {sig_label}\n"
                f"  Effective Samples (n_eff): {c.get('n_eff_joint', 'N/A')}"
            )
        correlation_summary = "\n\n".join(items)

    state = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": current_price_f,
        "indicators": indicator_values,
        "top_correlations": top_correlations,
        "indicator_summary": indicator_summary,
        "correlation_summary": correlation_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("State built", symbol=symbol, timeframe=timeframe, current_price=current_price)
    return state


async def _fetch_top_correlations(symbol: str, limit: int = 3) -> list[dict]:
    try:
        rows = _corr_conn.execute(
            "SELECT DISTINCT ON (pair) timestamp, pair, anchor, alt, timeframe, "
            "coefficient, dominant_lag, direction, p_value, n_eff_joint "
            "FROM correlation_snapshot "
            "WHERE (anchor = ? OR alt = ?) AND significant = 1 "
            "ORDER BY pair, timestamp DESC",
            [symbol, symbol],
        ).fetchall()
        logger.debug("Correlation raw rows", symbol=symbol, count=len(rows), sample=rows[:2] if rows else None)
    except Exception as e:
        logger.error("Failed to query correlation matrix", symbol=symbol, error=str(e))
        return []

    results = []
    for row in rows:
        results.append({
            "timestamp": row[0],
            "pair": row[1],
            "anchor": row[2],
            "alt": row[3],
            "timeframe": row[4],
            "coefficient": row[5],
            "dominant_lag": row[6],
            "direction": row[7],
            "p_value": row[8],
            "n_eff_joint": row[9],
        })

    results.sort(key=lambda x: x["p_value"])
    return results[:limit]


def _parse_timeframe(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 1440
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")
