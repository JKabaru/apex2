from __future__ import annotations

import math
from statistics import mean, median_low, stdev
from typing import Sequence


def compute_median(values: Sequence[float]) -> float:
    sorted_vals = sorted(values)
    if not sorted_vals:
        return 0.0
    return median_low(sorted_vals)


def compute_percentile(values: Sequence[float], p: float) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def compute_wilson_interval(
    wins: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p_hat = wins / total
    denominator = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total)
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_sharpe_ratio(returns: Sequence[float], rfr: float = 0.0) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean_ret = mean(returns)
    std_ret = stdev(returns)
    if std_ret == 0.0:
        return 0.0
    return math.sqrt(n) * (mean_ret - rfr) / std_ret


def compute_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return abs(gross_profit / gross_loss)


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd
