from __future__ import annotations

import math

import pytest

from src.intelligence.policies import OutlierPolicy
from src.intelligence.statistics import RobustCalculator


def test_distribution_known_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    dist = RobustCalculator.compute_distribution(values)
    assert dist["count"] == 10
    assert dist["median"] == 5.5
    assert abs(dist["mean"] - 5.5) < 1e-9
    assert dist["iqr"] == 4.5
    assert abs(dist["p10"] - 1.9) < 1e-9
    assert abs(dist["p90"] - 9.1) < 1e-9
    cv = dist["cv"]
    expected_cv = math.sqrt(8.25) / 5.5
    assert abs(cv - expected_cv) < 1e-9


def test_distribution_empty():
    dist = RobustCalculator.compute_distribution([])
    assert dist["count"] == 0
    assert dist["median"] == 0.0
    assert dist["mean"] == 0.0
    assert dist["p10"] == 0.0
    assert dist["p90"] == 0.0
    assert dist["iqr"] == 0.0
    assert dist["cv"] == 0.0


def test_distribution_single():
    dist = RobustCalculator.compute_distribution([42.0])
    assert dist["count"] == 1
    assert dist["median"] == 42.0
    assert dist["p10"] == 42.0
    assert dist["p90"] == 42.0
    assert dist["iqr"] == 0.0


def test_distribution_two_values():
    dist = RobustCalculator.compute_distribution([0.0, 10.0])
    assert dist["count"] == 2
    assert dist["median"] == 5.0
    assert dist["p10"] == 1.0
    assert dist["p90"] == 9.0


def test_outlier_removal_iqr():
    policy = OutlierPolicy(method="IQR", multiplier=1.5, minimum_samples=5)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    cleaned, outliers = RobustCalculator.apply_outlier_policy(values, policy)
    assert outliers == 1
    assert 100.0 not in cleaned
    assert all(v in cleaned for v in [1.0, 2.0, 3.0, 4.0, 5.0])


def test_outlier_removal_no_outliers():
    policy = OutlierPolicy(method="IQR", multiplier=1.5, minimum_samples=5)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cleaned, outliers = RobustCalculator.apply_outlier_policy(values, policy)
    assert outliers == 0
    assert len(cleaned) == 6


def test_outlier_removal_below_minimum():
    policy = OutlierPolicy(method="IQR", multiplier=1.5, minimum_samples=10)
    values = [1.0, 2.0, 3.0, 5.0, 1000.0]
    cleaned, outliers = RobustCalculator.apply_outlier_policy(values, policy)
    assert outliers == 0  # skipped because n < minimum_samples
    assert len(cleaned) == 5


def test_outlier_removal_mad():
    policy = OutlierPolicy(method="MAD", multiplier=3.0, minimum_samples=5)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    cleaned, outliers = RobustCalculator.apply_outlier_policy(values, policy)
    assert outliers == 1
    assert 100.0 not in cleaned


def test_constant_values_have_zero_cv():
    dist = RobustCalculator.compute_distribution([5.0, 5.0, 5.0, 5.0, 5.0])
    assert dist["cv"] == 0.0
    assert dist["median"] == 5.0
    assert dist["iqr"] == 0.0


def test_negative_values():
    dist = RobustCalculator.compute_distribution([-10.0, -5.0, 0.0, 5.0, 10.0])
    assert dist["median"] == 0.0
    assert dist["count"] == 5
    assert dist["iqr"] > 0.0
