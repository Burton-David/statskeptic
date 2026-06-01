"""Degenerate-input behavior. The rule under test: statskeptic never crashes with a raw
scipy/numpy error and never returns a fabricated number. It either declines, raises a
clean AnalysisError naming the problem, or rejects bad parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statskeptic import PlanHints, Verdict, analyze
from statskeptic.cli import main
from statskeptic.errors import AnalysisError
from statskeptic.profile import ColumnKind, build_profile
from statskeptic.stats import (
    anova_oneway,
    chi_square,
    logistic,
    mann_whitney_u,
    ols,
    pearson,
    students_t,
)

# --- routine-level guards: degenerate input raises a clean AnalysisError ---


def test_t_test_rejects_a_singleton_group():
    with pytest.raises(AnalysisError, match="2 observations per group"):
        students_t(np.array([1.0]), np.array([2.0, 3.0, 4.0]))


def test_t_test_rejects_two_constant_groups():
    with pytest.raises(AnalysisError, match="constant"):
        students_t(np.array([5.0, 5, 5]), np.array([5.0, 5, 5]))


def test_mann_whitney_rejects_all_identical():
    with pytest.raises(AnalysisError, match="identical"):
        mann_whitney_u(np.array([3.0, 3, 3]), np.array([3.0, 3, 3]))


def test_pearson_rejects_too_few_points():
    with pytest.raises(AnalysisError, match="3 paired"):
        pearson(np.array([1.0, 2]), np.array([3.0, 4]))


def test_pearson_rejects_constant_variable():
    with pytest.raises(AnalysisError, match="constant"):
        pearson(np.array([2.0, 2, 2, 2]), np.array([1.0, 2, 3, 4]))


def test_chi_square_rejects_degenerate_table():
    with pytest.raises(AnalysisError, match="2 rows and 2 columns"):
        chi_square(np.array([[1, 2, 3]]))
    with pytest.raises(AnalysisError, match="all zeros"):
        chi_square(np.array([[0, 0], [5, 7]]))


def test_anova_rejects_too_few_observations():
    with pytest.raises(AnalysisError, match="not enough observations"):
        anova_oneway(
            [np.array([1.0]), np.array([2.0]), np.array([3.0])], ["a", "b", "c"]
        )


def test_ols_rejects_constant_outcome():
    x = np.arange(20.0).reshape(-1, 1)
    with pytest.raises(AnalysisError, match="constant"):
        ols(np.full(20, 7.0), x, ["x"])


def test_ols_rejects_singular_design():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 30)
    design = np.column_stack([x, x])  # x2 == x1
    with pytest.raises(AnalysisError, match="collinear"):
        ols(rng.normal(0, 1, 30), design, ["x1", "x2"])


def test_logistic_rejects_single_class():
    rng = np.random.default_rng(0)
    with pytest.raises(AnalysisError, match="one class"):
        logistic(np.zeros(20), rng.normal(0, 1, 20).reshape(-1, 1), ["x"])


def test_infinities_are_treated_as_missing_not_crashed_on():
    # Regression for a real corpus file: inf in both groups made inf - inf = nan blow up
    # the Hodges-Lehmann estimator. Infinities are now dropped like NaNs.
    a = np.array([1.0, 2, 3, np.inf, 5, 6])
    b = np.array([2.0, 4, np.inf, 8, 10, 12])
    r = mann_whitney_u(a, b)
    assert r.n == 10  # the two infinities were dropped

    df = pd.DataFrame(
        {
            "arm": ["a"] * 10 + ["b"] * 10,
            "score": np.concatenate(
                [np.r_[np.full(1, np.inf), np.arange(9.0)], np.arange(10.0, 20.0)]
            ),
        }
    )
    report = analyze(df, "Does score differ by arm?")
    assert report.analyses[0].result.n == 19  # one inf dropped, no crash


# --- agent-level: the whole pipeline handles bad data without lying ---


def test_empty_frame_declines():
    assert analyze(pd.DataFrame(), "Does a differ from b?").verdict == Verdict.declined


def test_all_missing_outcome_declines():
    df = pd.DataFrame({"grp": ["a", "b"] * 10, "y": [np.nan] * 20})
    assert analyze(df, "Does y differ by grp?").verdict == Verdict.declined


def test_text_outcome_via_hint_raises_cleanly():
    df = pd.DataFrame({"grp": ["a", "b"] * 10, "y": ["lo", "hi"] * 10})
    with pytest.raises(AnalysisError, match="not numeric"):
        analyze(df, "Does y differ by grp?", hints=PlanHints(outcome="y", group="grp"))


def test_constant_outcome_via_hint_raises_cleanly():
    df = pd.DataFrame({"grp": ["a", "b"] * 10, "y": [5.0] * 20})
    with pytest.raises(AnalysisError):
        analyze(df, "Does y differ by grp?", hints=PlanHints(outcome="y", group="grp"))


def test_multicategory_outcome_is_declined_not_run_as_ols():
    # Regression for a real bug: a 3-level coded outcome was silently regressed with OLS.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 30), "y": [0, 1, 2] * 10})
    r = analyze(df, "Predict y", hints=PlanHints(outcome="y", predictors=["x"]))
    assert r.verdict == Verdict.declined
    assert "categories" in r.declined.reason


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_alpha_out_of_range_is_rejected(bad):
    df = pd.DataFrame({"grp": ["a", "b"] * 10, "y": np.arange(20.0)})
    with pytest.raises(ValueError, match="alpha"):
        analyze(df, "Does y differ by grp?", alpha=bad)


# --- profiler: degenerate columns characterize without crashing ---


def test_profiler_handles_all_nan_and_single_value_columns():
    # 'x' has many distinct values, so it stays numeric (a handful of integer codes
    # would be read as a categorical factor instead, which is the intended heuristic).
    df = pd.DataFrame({"empty": [np.nan] * 30, "const": [4] * 30, "x": np.arange(30.0)})
    prof = build_profile(df)
    kinds = {c.name: c.kind for c in prof.columns}
    assert kinds["empty"] == ColumnKind.constant
    assert kinds["const"] == ColumnKind.constant
    assert prof.by_name("x").numeric is not None


# --- CLI robustness ---


def test_cli_empty_file_is_usage_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert main(["analyze", str(path), "-q", "anything", "--quiet"]) == 64


def test_cli_unknown_outcome_declines(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"grp": ["a", "b"] * 10, "y": np.arange(20.0)}).to_csv(
        path, index=False
    )
    code = main(
        [
            "analyze",
            str(path),
            "-q",
            "Does y differ by grp?",
            "--outcome",
            "nope",
            "--quiet",
        ]
    )
    assert code == 3


def test_cli_degenerate_data_is_internal_error(tmp_path, capsys):
    path = tmp_path / "const.csv"
    pd.DataFrame({"grp": ["a", "b"] * 10, "y": [5.0] * 20}).to_csv(path, index=False)
    code = main(
        [
            "analyze",
            str(path),
            "-q",
            "Does y differ by grp?",
            "--outcome",
            "y",
            "--quiet",
        ]
    )
    assert code == 70
    assert "analysis failed" in capsys.readouterr().err


def test_cli_parses_dates_from_csv_for_trend(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60).strftime("%Y-%m-%d"),
            "revenue": np.linspace(100, 200, 60) + rng.normal(0, 5, 60),
        }
    )
    path = tmp_path / "trend.csv"
    df.to_csv(path, index=False)
    code = main(
        [
            "analyze",
            str(path),
            "-q",
            "Is there a trend in revenue over time?",
            "--quiet",
        ]
    )
    assert code in (0, 2)  # a real trend analysis ran, not a decline (3) or crash
