from __future__ import annotations

from src.intelligence.policies import OutlierPolicy


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile. Pure Python, no numpy."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    k = p * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_vals[f] * (1.0 - c) + sorted_vals[f + 1] * c
    return sorted_vals[f]


class RobustCalculator:
    """Domain-agnostic math engine. Operates ONLY on list[float].
    Never references trading field names."""

    @staticmethod
    def compute_distribution(values: list[float]) -> dict:
        """Return median, mean, p10, p90, iqr, cv, count.

        All metrics are robust / resistant to skewed distributions.
        Uses pure Python sorting — no numpy.
        """
        n = len(values)
        if n == 0:
            return {
                "median": 0.0,
                "mean": 0.0,
                "p10": 0.0,
                "p90": 0.0,
                "iqr": 0.0,
                "cv": 0.0,
                "count": 0,
            }

        sorted_vals = sorted(values)

        median = _percentile(sorted_vals, 0.5)
        p10 = _percentile(sorted_vals, 0.1)
        p90 = _percentile(sorted_vals, 0.9)
        q1 = _percentile(sorted_vals, 0.25)
        q3 = _percentile(sorted_vals, 0.75)
        iqr = q3 - q1
        mean = sum(sorted_vals) / n
        variance = sum((x - mean) ** 2 for x in sorted_vals) / n
        std = variance ** 0.5
        cv = std / mean if abs(mean) > 1e-9 else 0.0

        return {
            "median": median,
            "mean": mean,
            "p10": p10,
            "p90": p90,
            "iqr": iqr,
            "cv": cv,
            "count": n,
        }

    @staticmethod
    def apply_outlier_policy(
        values: list[float],
        policy: OutlierPolicy,
    ) -> tuple[list[float], int]:
        """Remove outliers using IQR or MAD method.

        Returns (cleaned_values, outlier_count).
        Policy's minimum_samples is respected — no filtering below that threshold.
        """
        n = len(values)
        if n < policy.minimum_samples:
            return list(values), 0

        sorted_vals = sorted(values)

        if policy.method == "IQR":
            q1 = _percentile(sorted_vals, 0.25)
            q3 = _percentile(sorted_vals, 0.75)
            spread = q3 - q1
            lower = q1 - policy.multiplier * spread
            upper = q3 + policy.multiplier * spread
        elif policy.method == "MAD":
            median = _percentile(sorted_vals, 0.5)
            abs_devs = sorted(abs(v - median) for v in values)
            mad = _percentile(abs_devs, 0.5)
            lower = median - policy.multiplier * mad
            upper = median + policy.multiplier * mad
        else:
            raise ValueError(f"Unknown outlier method: {policy.method}")

        cleaned = [v for v in values if lower <= v <= upper]
        outlier_count = len(values) - len(cleaned)
        return cleaned, outlier_count
