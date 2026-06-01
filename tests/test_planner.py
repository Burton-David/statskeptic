from __future__ import annotations

import numpy as np
import pandas as pd

from statskeptic.plan import Decline, Method, PlanHints, QuestionType, plan
from statskeptic.profile import build_profile


def _profile(df: pd.DataFrame):
    return build_profile(df)


def test_two_group_comparison_defaults_to_students_t():
    df = pd.DataFrame({"treatment": ["a", "b"] * 20, "recovery": np.arange(40.0)})
    p = plan("Does treatment change recovery?", _profile(df))
    assert not isinstance(p, Decline)
    assert p.question_type == QuestionType.comparison
    # The naive default is Student's t; the critique is what upgrades it later.
    assert p.method == Method.students_t
    assert p.outcome == "recovery"
    assert p.group == "treatment"


def test_three_groups_route_to_anova():
    df = pd.DataFrame({"site": ["x", "y", "z"] * 20, "score": np.arange(60.0)})
    p = plan("Does score differ across site?", _profile(df))
    assert not isinstance(p, Decline)
    assert p.method == Method.anova_oneway


def test_two_numeric_association_is_pearson():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"height": rng.normal(0, 1, 50), "weight": rng.normal(0, 1, 50)})
    p = plan("Is there a relationship between height and weight?", _profile(df))
    assert not isinstance(p, Decline)
    assert p.method == Method.pearson


def test_screen_question_fans_out():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"v{i}": rng.normal(0, 1, 40) for i in range(5)})
    df["outcome"] = rng.normal(0, 1, 40)
    p = plan("Which variables are associated with the outcome?", _profile(df))
    assert not isinstance(p, Decline)
    assert p.question_type == QuestionType.screen
    assert len(p.candidates) == 5


def test_regression_question_uses_ols():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"bp": rng.normal(0, 1, 50), "age": rng.normal(0, 1, 50)})
    p = plan("Predict bp from age", _profile(df), PlanHints(outcome="bp"))
    assert not isinstance(p, Decline)
    assert p.method == Method.ols


def test_unmappable_question_is_declined():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    p = plan("Forecast next quarter with a neural network", _profile(df))
    assert isinstance(p, Decline)
    assert p.supported


def test_comparison_of_two_categoricals_routes_to_chi_square():
    # "differ" is a comparison cue, but with no numeric outcome and two categoricals the
    # honest method is chi-square, not a t-test.
    df = pd.DataFrame(
        {"smoker": ["y", "n"] * 30, "diagnosis": ["pos", "neg", "neg", "pos"] * 15}
    )
    p = plan("Do smokers differ from non-smokers in diagnosis?", _profile(df))
    assert not isinstance(p, Decline)
    assert p.method == Method.chi_square


def test_association_declines_when_columns_cannot_be_resolved():
    # An association question with only one usable numeric column has no second variable.
    df = pd.DataFrame({"height": np.arange(30.0), "label": ["x"] * 15 + ["yy"] * 15})
    p = plan("Is there a correlation in this data?", _profile(df))
    assert isinstance(p, Decline)


def test_hints_override_column_matching():
    df = pd.DataFrame(
        {"grp": ["a", "b"] * 20, "m1": np.arange(40.0), "m2": np.arange(40.0)}
    )
    # Two numeric columns: name the outcome explicitly to disambiguate.
    p = plan(
        "Does the group affect the measurement?", _profile(df), PlanHints(outcome="m2")
    )
    assert not isinstance(p, Decline)
    assert p.outcome == "m2"
