from collections import deque

import duckdb
import numpy as np
import structlog
from scipy.stats import t as student_t

logger = structlog.get_logger("correlation_engine")

OHLCV_DB = "data/ohlcv.duckdb"





def _compute_acf(z: np.ndarray, max_p: int) -> np.ndarray:
    z_c = z - np.mean(z)
    denom = z_c @ z_c
    if denom <= 1e-12 or max_p < 1:
        return np.zeros(max_p)
    n = len(z_c)
    c = np.correlate(z_c, z_c, mode="full")
    max_p_actual = min(max_p, n - 1)
    result = np.zeros(max_p)
    result[:max_p_actual] = c[n : n + max_p_actual] / denom
    return result


def compute_lagged_cross_correlation(
    anchor_returns: np.ndarray,
    alt_returns: np.ndarray,
    max_lag: int,
    base_half_life: float,
    window_size: int = 500,
    acf_truncation_lag: int = 10,
    alpha_crit: float = 0.01,
    min_half_life: float = 15,
    max_half_life: float = 180,
) -> dict:
    total_needed = window_size + max_lag
    if len(anchor_returns) < total_needed or len(alt_returns) < window_size:
        return {"status": "insufficient_data"}

    y = alt_returns[-window_size:]
    x_padded = anchor_returns[-total_needed:]

    x_windows = np.lib.stride_tricks.sliding_window_view(x_padded, window_size)
    lagged_matrix = x_windows[:max_lag][::-1]

    sigma_short = np.std(anchor_returns[-30:])
    sigma_long = np.std(anchor_returns[-window_size:])
    phi_t = sigma_short / (sigma_long + 1e-12)

    dynamic_half_life = np.clip(
        base_half_life / (phi_t ** 1.0), min_half_life, max_half_life
    )
    lam = np.log(2) / dynamic_half_life

    w = np.exp(-lam * np.arange(window_size)[::-1])
    w_norm = w / np.sum(w)

    mu_x = np.dot(lagged_matrix, w_norm)
    mu_y = np.dot(y, w_norm)

    x_centered = lagged_matrix - mu_x[:, np.newaxis]
    y_centered = y - mu_y

    var_x = np.dot(x_centered ** 2, w_norm)
    var_y = np.dot(y_centered ** 2, w_norm)

    if var_y <= 1e-12:
        return {"status": "zero_variance_alt"}

    cov_xy = np.dot(x_centered, y_centered * w_norm)
    corr_vector = cov_xy / (np.sqrt(var_x * var_y) + 1e-15)

    acf_x = _compute_acf(anchor_returns[-window_size:], acf_truncation_lag)
    acf_y = _compute_acf(y, acf_truncation_lag)

    C = max(1.0 + 2.0 * np.sum(acf_x * acf_y), 0.1)
    n_eff_kish = 1.0 / np.sum(w_norm ** 2)
    n_eff_joint = n_eff_kish / C

    if n_eff_joint <= 2.1:
        return {"status": "insufficient_edf"}

    corr_clipped = np.clip(corr_vector, -0.999999, 0.999999)
    t_stats = corr_clipped * np.sqrt(
        (n_eff_joint - 2) / (1.0 - corr_clipped ** 2)
    )
    p_values = student_t.sf(np.abs(t_stats), df=n_eff_joint - 2) * 2

    abs_corr = np.abs(corr_vector)
    dom_idx = np.argmax(abs_corr)
    dom_lag = int(dom_idx + 1)
    dom_coef = float(corr_vector[dom_idx])
    dom_p_value = float(p_values[dom_idx])
    significant = bool(dom_p_value < alpha_crit)
    direction = 1 if dom_coef > 0 else -1

    return {
        "dominant_lag": dom_lag if significant else -1,
        "coefficient": dom_coef,
        "direction": direction,
        "p_value": dom_p_value,
        "significant": significant,
        "n_eff_joint": float(n_eff_joint),
        "dynamic_half_life": float(dynamic_half_life),
        "status": "success",
    }


class CorrelationEngine:
    def __init__(
        self,
        rolling_window_candles: int = 500,
        max_lag: int = 15,
        base_half_life: float = 60.0,
        min_half_life: float = 15.0,
        max_half_life: float = 180.0,
        acf_truncation_lag: int = 10,
        alpha_crit: float = 0.01,
        update_buffer_candles: int = 10,
        anchors: list | None = None,
        alternates: list | None = None,
        max_timeframe_m: int = 1440,
        db_path: str = OHLCV_DB,
    ):
        self._window_size = rolling_window_candles
        self._max_lag = max_lag
        self._base_half_life = base_half_life
        self._min_half_life = min_half_life
        self._max_half_life = max_half_life
        self._acf_truncation_lag = acf_truncation_lag
        self._alpha_crit = alpha_crit
        self._update_buffer_candles = update_buffer_candles
        self._anchors = anchors or []
        self._alternates = alternates or []
        self._active_timeframes = list(range(1, max_timeframe_m + 1))
        self._db_path = db_path

        maxlen = self._window_size + self._max_lag
        self._buffers: dict[str, deque] = {}
        self._last_close: dict[str, float] = {}
        for sym in self._anchors + self._alternates:
            self._buffers[sym] = deque(maxlen=maxlen)
            self._last_close[sym] = 0.0

        self._ingested_counter = 0
        self._ohlcv_conn = duckdb.connect(self._db_path)

    def close(self):
        self._ohlcv_conn.close()

    def seed_history(self, symbol: str, closes: list[float]):
        if not closes or len(closes) < 2:
            return
        if symbol not in self._buffers:
            maxlen = self._window_size + self._max_lag
            self._buffers[symbol] = deque(maxlen=maxlen)
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                log_returns.append(float(np.log(closes[i] / closes[i - 1])))
        self._buffers[symbol].extend(log_returns)
        self._last_close[symbol] = closes[-1]
        logger.info(
            "Seeded correlation buffer",
            symbol=symbol,
            log_returns=len(log_returns),
            last_close=closes[-1],
        )

    def update_price(self, symbol: str, close: float) -> float | None:
        prev = self._last_close.get(symbol)
        self._last_close[symbol] = close
        if prev is None or prev <= 0 or close <= 0:
            return None
        return float(np.log(close / prev))

    def append_log_return(self, symbol: str, log_return: float):
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(
                maxlen=self._window_size + self._max_lag
            )
        self._buffers[symbol].append(log_return)
        self._ingested_counter += 1

    def ready_to_compute(self) -> bool:
        return self._ingested_counter >= self._update_buffer_candles

    def _fetch_returns(self, tf_minutes: int, symbol: str) -> np.ndarray:
        limit = self._window_size + self._max_lag + 1
        if tf_minutes == 1:
            rows = self._ohlcv_conn.execute(
                "SELECT close FROM ohlcv_1m WHERE symbol = ? "
                "ORDER BY open_time DESC LIMIT ?",
                [symbol, limit],
            ).fetchall()
        else:
            try:
                rows = self._ohlcv_conn.execute(
                    "SELECT close FROM ohlcv_agg WHERE tf_minutes = ? AND symbol = ? "
                    "ORDER BY bucket DESC LIMIT ?",
                    [tf_minutes, symbol, limit],
                ).fetchall()
            except Exception:
                return np.array([])

        if len(rows) < 2:
            return np.array([])
        closes = np.array([r[0] for r in reversed(rows)])
        return np.log(closes[1:] / closes[:-1])

    def compute_all_pairs(self) -> list[dict]:
        results = []

        for tf_minutes in self._active_timeframes:
            for anchor in self._anchors:
                for alt in self._alternates:
                    if anchor == alt:
                        continue

                    if tf_minutes == 1:
                        anchor_arr = np.array(
                            self._buffers.get(anchor, []), dtype=float
                        )
                        alt_arr = np.array(
                            self._buffers.get(alt, []), dtype=float
                        )
                    else:
                        anchor_arr = self._fetch_returns(tf_minutes, anchor)
                        alt_arr = self._fetch_returns(tf_minutes, alt)

                    if len(anchor_arr) < self._window_size + self._max_lag or \
                       len(alt_arr) < self._window_size:
                        continue

                    try:
                        result = compute_lagged_cross_correlation(
                            anchor_returns=anchor_arr,
                            alt_returns=alt_arr,
                            max_lag=self._max_lag,
                            base_half_life=self._base_half_life,
                            window_size=self._window_size,
                            acf_truncation_lag=self._acf_truncation_lag,
                            alpha_crit=self._alpha_crit,
                            min_half_life=self._min_half_life,
                            max_half_life=self._max_half_life,
                        )
                    except Exception as e:
                        logger.error(
                            "LEW-CCF computation failed",
                            pair=f"{anchor}/{alt}",
                            timeframe=tf_minutes,
                            error=str(e),
                        )
                        result = {"status": "error"}
                    result["pair"] = f"{anchor}/{alt}"
                    result["anchor"] = anchor
                    result["alt"] = alt
                    result["timeframe"] = tf_minutes
                    results.append(result)

        self._ingested_counter = 0
        return results
