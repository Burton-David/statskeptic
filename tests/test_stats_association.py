"""Known-answer tests for association routines."""

from __future__ import annotations

import numpy as np
import pytest

from statskeptic.stats import chi_square, fisher_exact, pearson, spearman


def test_pearson_matches_hand_computation():
    # x=[1..5], y=[2,4,5,4,5]. dx=[-2,-1,0,1,2], dy=[-2,0,1,0,1].
    # Sxy=6, Sxx=10, Syy=6 -> r = 6/sqrt(60) = 0.774597.
    x = np.array([1, 2, 3, 4, 5.0])
    y = np.array([2, 4, 5, 4, 5.0])
    r = pearson(x, y)
    assert r.statistic == pytest.approx(0.774597, rel=1e-5)
    assert r.ci is not None
    assert r.ci.low < r.statistic < r.ci.high


def test_spearman_with_a_swap():
    # x ranks 1..4, y=[1,3,2,4] ranks [1,3,2,4]. sum d^2 = 0+1+1+0 = 2.
    # rho = 1 - 6*2/(4*15) = 0.8.
    x = np.array([1, 2, 3, 4.0])
    y = np.array([1, 3, 2, 4.0])
    assert spearman(x, y).statistic == pytest.approx(0.8, rel=1e-9)


def test_spearman_perfect_monotonic_has_no_ci():
    # y = x^2 is perfectly monotonic, so rho = 1 and the Fisher-z CI is undefined; we
    # report None rather than a fabricated interval.
    x = np.array([1, 2, 3, 4, 5.0])
    y = x**2
    r = spearman(x, y)
    assert r.statistic == pytest.approx(1.0)
    assert r.ci is None


def test_chi_square_2x2_with_yates():
    # [[10,20],[30,40]]: expected [[12,18],[28,42]], all |O-E|=2. Yates chi2 =
    # sum (1.5^2)/E = 2.25*(1/12+1/18+1/28+1/42) = 0.44643. Cramer's V = sqrt(chi2/100).
    table = np.array([[10, 20], [30, 40]])
    r = chi_square(table)
    assert r.statistic == pytest.approx(0.44643, abs=1e-4)
    assert r.dof == 1
    assert r.effect.value == pytest.approx(np.sqrt(0.44643 / 100), abs=1e-4)


def test_chi_square_flags_small_expected_counts():
    # A sparse table where an expected count falls below 5.
    table = np.array([[1, 1], [1, 20]])
    r = chi_square(table)
    counts = [c for c in r.assumptions if c.name == "expected_cell_counts"]
    assert counts and not counts[0].holds


def test_fisher_exact_odds_ratio():
    # [[8,2],[1,5]]: sample odds ratio = (8*5)/(2*1) = 20.
    table = np.array([[8, 2], [1, 5]])
    r = fisher_exact(table)
    assert r.effect.value == pytest.approx(20.0, rel=1e-6)
    assert r.method == "Fisher's exact test"
