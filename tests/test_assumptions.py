from __future__ import annotations

import numpy as np

from statskeptic.stats.assumptions import (
    check_equal_variance,
    check_expected_cell_counts,
    check_normality,
)


def test_normality_passes_on_clean_normal_data():
    rng = np.random.default_rng(0)
    c = check_normality(rng.normal(0, 1, 300), "x")
    assert c.holds


def test_normality_fails_high_severity_on_heavy_skew():
    rng = np.random.default_rng(0)
    # Lognormal at moderate n: skew well past 1, so a mean-based test is the wrong tool
    # and the severity must be high enough to force a method switch downstream.
    c = check_normality(rng.lognormal(0, 1, 60), "x")
    assert not c.holds
    assert c.severity.value == "high"


def test_normality_severity_drops_when_skew_is_mild():
    # A mild, near-symmetric deviation at a healthy n leans on the CLT, so even if
    # Shapiro rejects, the consequence is low, not high.
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 400)
    c = check_normality(x, "x")
    assert c.severity.value in ("low", "medium")


def test_equal_variance_flags_unequal_spread():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0, 5, 50)
    c = check_equal_variance([a, b])
    assert not c.holds
    assert c.severity.value == "high"


def test_expected_counts_rule():
    good = check_expected_cell_counts(np.array([[20, 20], [20, 20.0]]))
    assert good.holds
    sparse = check_expected_cell_counts(np.array([[0.5, 0.5], [10, 10.0]]))
    assert not sparse.holds


def test_normality_skips_shapiro_above_the_cap_and_judges_by_skew():
    # Above ~5000 Shapiro over-rejects, so we report no W and decide on skew instead:
    # clean data passes at low severity, heavy skew still fails at high severity.
    rng = np.random.default_rng(0)
    big_normal = check_normality(rng.normal(0, 1, 6000), "x")
    assert big_normal.holds and big_normal.statistic is None
    assert big_normal.severity.value == "low"

    big_skewed = check_normality(rng.lognormal(0, 1, 6000), "x")
    assert not big_skewed.holds and big_skewed.statistic is None
    assert big_skewed.severity.value == "high"


def test_normality_is_not_assessable_below_three():
    c = check_normality(np.array([1.0, 2.0]), "x")
    assert c.holds and c.severity.value == "info"
