"""Coverage for the rendering and planner branches the trap tests do not reach:
regression/screen/decline markdown, plus trend and chi-square planning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statskeptic import PlanHints, analyze
from statskeptic.plan import Decline, Method, QuestionType, plan
from statskeptic.profile import build_profile


def test_regression_report_renders_coefficient_table():
    rng = np.random.default_rng(0)
    n = 120
    age = rng.normal(50, 10, n)
    weight = rng.normal(70, 12, n)
    df = pd.DataFrame(
        {
            "age": age,
            "weight": weight,
            "bp": 0.4 * age + 0.2 * weight + rng.normal(0, 5, n),
        }
    )
    report = analyze(
        df, "Predict bp", hints=PlanHints(outcome="bp", predictors=["age", "weight"])
    )
    text = report.explain()
    assert "Coefficients (coefficient)" in text
    assert "age" in text and "weight" in text


def test_logistic_report_renders_odds_ratios():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.5 + 0.8 * x)))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"x": x, "converted": y})
    report = analyze(
        df, "Predict converted", hints=PlanHints(outcome="converted", predictors=["x"])
    )
    assert "odds ratio" in report.explain()


def test_screen_report_lists_corrected_columns(screen_many):
    text = analyze(
        screen_many, "Which variables are associated with the outcome?"
    ).explain()
    assert "Screen across candidate variables" in text
    assert "corrected p" in text


def test_declined_report_lists_supported_methods():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    text = analyze(df, "Cluster these into segments").explain()
    assert "## Declined" in text
    assert "statskeptic can answer" in text


def test_trend_plan_uses_time_as_predictor():
    df = pd.DataFrame(
        {
            "day": pd.to_datetime("2026-01-01") + pd.to_timedelta(range(60), unit="D"),
            "revenue": np.linspace(100, 200, 60),
        }
    )
    p = plan("Is there a trend in revenue over time?", build_profile(df))
    assert not isinstance(p, Decline)
    assert p.question_type == QuestionType.trend
    assert p.method == Method.ols
    assert p.predictors == ["day"]


def test_chi_square_for_two_categoricals():
    df = pd.DataFrame(
        {"smoker": ["yes", "no"] * 40, "diagnosis": ["pos", "neg", "neg", "pos"] * 20}
    )
    p = plan("Is smoker related to diagnosis?", build_profile(df))
    assert not isinstance(p, Decline)
    assert p.method == Method.chi_square
