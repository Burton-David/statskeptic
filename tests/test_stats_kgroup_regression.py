"""Known-answer tests for the k-group and regression routines."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from statskeptic.errors import AnalysisError
from statskeptic.stats import anova_oneway, kruskal, logistic, ols


def test_anova_matches_hand_computed_sums_of_squares():
    # g1=[1..4] mean 2.5, g2=[2..5] mean 3.5, g3=[5..8] mean 6.5, grand mean 4.1667.
    # SS_between = 4*((2.5-4.1667)^2+(3.5-4.1667)^2+(6.5-4.1667)^2) = 34.667.
    # SS_within = 15 (each group contributes 5). F = (34.667/2)/(15/9) = 10.4.
    g = [np.array([1, 2, 3, 4.0]), np.array([2, 3, 4, 5.0]), np.array([5, 6, 7, 8.0])]
    r = anova_oneway(g, ["a", "b", "c"])
    assert r.statistic == pytest.approx(10.4, rel=1e-6)
    assert r.dof_between == 2
    assert r.dof_within == 9
    # eta-squared = 34.667 / 49.667 = 0.69799.
    assert r.effect.value == pytest.approx(0.69799, abs=1e-4)


def test_kruskal_epsilon_squared_follows_from_h():
    g = [np.array([1, 2, 3, 4.0]), np.array([2, 3, 4, 5.0]), np.array([5, 6, 7, 8.0])]
    r = kruskal(g, ["a", "b", "c"])
    # epsilon^2 = (H - k + 1)/(n - k) with k=3, n=12 (Tomczak & Tomczak 2014).
    expected = (r.statistic - 3 + 1) / (12 - 3)
    assert r.effect.value == pytest.approx(expected, rel=1e-9)


def test_ols_recovers_a_known_line():
    # y = 2x + 1 exactly -> slope 2, intercept 1, R^2 = 1.
    x = np.array([1, 2, 3, 4, 5.0]).reshape(-1, 1)
    y = np.array([3, 5, 7, 9, 11.0])
    r = ols(y, x, ["x"])
    assert r.params["x"] == pytest.approx(2.0, rel=1e-9)
    assert r.params["const"] == pytest.approx(1.0, abs=1e-9)
    assert r.r_squared == pytest.approx(1.0, rel=1e-9)


def test_ols_matches_hand_computed_slope():
    # Same x,y as the Pearson test: slope = Sxy/Sxx = 6/10 = 0.6, intercept = 4 - 0.6*3
    # = 2.2, R^2 = r^2 = 0.6.
    x = np.array([1, 2, 3, 4, 5.0]).reshape(-1, 1)
    y = np.array([2, 4, 5, 4, 5.0])
    r = ols(y, x, ["x"])
    assert r.params["x"] == pytest.approx(0.6, rel=1e-6)
    assert r.params["const"] == pytest.approx(2.2, rel=1e-6)
    assert r.r_squared == pytest.approx(0.6, rel=1e-6)


def test_logistic_reports_odds_ratios_as_exp_of_coefficients():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(0, 1, n)
    # Overlapping classes so the MLE exists and converges cleanly.
    logit = -0.3 + 0.9 * x
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    r = logistic(y, x.reshape(-1, 1), ["x"])

    independent = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    assert r.params["x"] == pytest.approx(
        float(np.exp(independent.params[1])), rel=1e-6
    )
    assert r.converged
    assert all(c.holds for c in r.assumptions if c.name == "separation")


def test_logistic_flags_quasi_separation():
    # Almost separable: x=5 appears in both classes, the rest split cleanly. statsmodels
    # fits but fails to converge and warns, so we return a result flagged as unreliable.
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1.0])
    x = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9.0]).reshape(-1, 1)
    r = logistic(y, x, ["x"])
    assert not r.converged
    assert any(c.name == "separation" and not c.holds for c in r.assumptions)


def test_logistic_raises_on_perfect_separation():
    # x perfectly orders the outcome, so no MLE exists. Raising beats reporting a
    # fabricated odds ratio.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8.0]).reshape(-1, 1)
    with pytest.raises(AnalysisError, match="not identifiable"):
        logistic(y, x, ["x"])
