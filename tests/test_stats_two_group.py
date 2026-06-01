"""Known-answer tests for the two-group comparisons. Every expected number is either
hand-computed (arithmetic in the comment) or a value a reader can re-derive."""

from __future__ import annotations

import numpy as np
import pytest

from statskeptic.stats import mann_whitney_u, students_t, welch_t


def test_students_t_matches_hand_computation():
    # A=[1..5] mean 3, B=[3..7] mean 5, both variance 2.5. Mean diff -2, pooled SD
    # sqrt(2.5), SE = sqrt(2.5/5 + 2.5/5) = 1.0, so t = -2/1 = -2.0 on df=8.
    a = np.array([1, 2, 3, 4, 5.0])
    b = np.array([3, 4, 5, 6, 7.0])
    r = students_t(a, b)
    assert r.statistic == pytest.approx(-2.0, rel=1e-9)
    # Cohen's d = -2 / sqrt(2.5) = -1.264911.
    assert r.effect.value == pytest.approx(-1.264911, rel=1e-5)
    # 95% CI on the mean difference: -2 +/- t(.975, 8)=2.306 -> [-4.306, 0.306].
    assert r.ci is not None
    assert r.ci.low == pytest.approx(-4.306, abs=1e-3)
    assert r.ci.high == pytest.approx(0.306, abs=1e-3)


def test_welch_handles_unequal_variance():
    # A=[1,2,3] var 1, B=[10,20,30] var 100. Welch t = (2-20)/sqrt(1/3+100/3)
    # = -18 / sqrt(33.6667) = -3.1022.
    a = np.array([1, 2, 3.0])
    b = np.array([10, 20, 30.0])
    r = welch_t(a, b)
    assert r.statistic == pytest.approx(-3.1022, abs=1e-3)
    # With equal n the t-statistic is identical to Student's; what Welch changes is the
    # df, so it is more conservative. The p-value, not the t, is where they diverge.
    assert r.p_value > students_t(a, b).p_value


def test_mann_whitney_u_and_hodges_lehmann():
    # A entirely below B: U for A is 0, p is the smallest reachable for 3 vs 3 (0.1).
    a = np.array([1, 2, 3.0])
    b = np.array([4, 5, 6.0])
    r = mann_whitney_u(a, b)
    assert r.statistic == pytest.approx(0.0)
    assert r.p_value == pytest.approx(0.1, abs=1e-9)
    # rank-biserial r = 2*U/(n1*n2) - 1 = -1 when A never exceeds B.
    assert r.effect.value == pytest.approx(-1.0)
    # Hodges-Lehmann shift = median of the 9 pairwise diffs (sorted median is -3).
    assert r.hodges_lehmann == pytest.approx(-3.0)


def test_hodges_lehmann_even_count():
    # A=[1,2,3,4], B=[5,6,7,8]: 16 pairwise diffs, median = mean of the 8th and 9th
    # smallest = (-4, -4) = -4.
    a = np.array([1, 2, 3, 4.0])
    b = np.array([5, 6, 7, 8.0])
    assert mann_whitney_u(a, b).hodges_lehmann == pytest.approx(-4.0)


def test_skew_makes_normality_check_fail_at_high_severity():
    rng = np.random.default_rng(1)
    a = rng.lognormal(0, 1, 60)
    b = rng.lognormal(0.3, 1, 60)
    r = students_t(a, b)
    normality = [c for c in r.assumptions if c.name.startswith("normality")]
    assert any(not c.holds and c.severity.value == "high" for c in normality)
