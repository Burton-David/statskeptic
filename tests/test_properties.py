"""Property-based tests: Hypothesis throws nan, inf, ties, near-constants, and extreme
magnitudes at the routines and the agent, and checks the invariants that must hold for
any input. The contract is not that every input yields an answer, but that the answer is
never a fabricated or non-finite number and the failure is never an uncaught exception.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from statskeptic import PlanHints, Verdict, analyze
from statskeptic.errors import AnalysisError, StatskepticError
from statskeptic.stats import mann_whitney_u, pearson, spearman, students_t, welch_t
from statskeptic.stats.results import StatResult

# Bounded finite floats (the bound keeps products from overflowing into spurious
# infinities) unioned with the special values that must be handled: nan, +/-inf, zero.
_floats = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.sampled_from([float("nan"), float("inf"), float("-inf"), 0.0]),
)


def _finite_array(min_size: int):
    return arrays(
        np.float64,
        array_shapes(min_dims=1, max_dims=1, min_side=min_size, max_side=80),
        elements=_floats,
    )


def _check_result(r: StatResult) -> None:
    assert math.isfinite(r.statistic)
    assert math.isfinite(r.p_value) and 0.0 <= r.p_value <= 1.0
    assert math.isfinite(r.effect.value)
    if r.ci is not None:
        assert math.isfinite(r.ci.low) and math.isfinite(r.ci.high)
        assert r.ci.low <= r.ci.high
    assert r.computation.callable and r.computation.library_version


@given(a=_finite_array(1), b=_finite_array(1))
def test_two_group_routines_never_misbehave(a, b):
    for routine in (students_t, welch_t, mann_whitney_u):
        try:
            _check_result(routine(a, b))
        except AnalysisError:
            pass  # an honest refusal on degenerate data is allowed


@given(x=_finite_array(1), y=_finite_array(1))
def test_correlation_routines_stay_in_bounds(x, y):
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    for routine in (pearson, spearman):
        try:
            r = routine(x, y)
        except AnalysisError:
            continue
        _check_result(r)
        # A correlation coefficient is bounded; rounding can nudge it a hair past 1.
        assert -1.0001 <= r.statistic <= 1.0001


@given(
    n=st.integers(min_value=0, max_value=40),
    group=st.lists(st.sampled_from(["a", "b", "c"]), min_size=0, max_size=40),
    values=st.lists(_floats, min_size=0, max_size=40),
)
def test_analyze_on_random_frames_is_always_sound(n, group, values):
    size = min(len(group), len(values))
    df = pd.DataFrame({"grp": group[:size], "y": values[:size]})
    try:
        report = analyze(
            df, "Does y differ by grp?", hints=PlanHints(outcome="y", group="grp")
        )
    except (StatskepticError, ValueError):
        return  # honest refusal / bad data is acceptable
    assert report.verdict in set(Verdict)
    assert report.to_json() and report.explain()
    for analysis in report.analyses:
        _check_result(analysis.result)
