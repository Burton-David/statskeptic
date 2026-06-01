"""End-to-end: each planted trap, from question to verdict, through the public API."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statskeptic import PlanHints, Verdict, analyze


def test_skew_trap_is_caught_and_revised(skewed_two_group):
    report = analyze(skewed_two_group, "Does the treatment change recovery?")
    analysis = report.analyses[0]
    assert analysis.result.method == "Mann-Whitney U"
    assert any(s.trigger == "assumption.normality" for s in analysis.revision_steps)


def test_screen_applies_multiplicity_correction(screen_many):
    report = analyze(screen_many, "Which variables are associated with the outcome?")
    assert report.verdict == Verdict.defensible_with_caveats
    assert any(c.check_id == "mc.uncorrected" for c in report.session_critiques)
    raw = sum(a.result.p_value < 0.05 for a in report.analyses)
    corrected = sum(a.result.is_significant for a in report.analyses)
    assert corrected < raw  # the correction demoted at least one chance finding


def test_confounded_causal_question_cannot_conclude(confounded):
    report = analyze(confounded, "Does exercise cause better health?")
    assert report.verdict == Verdict.cannot_conclude
    assert any("causation" in line.lower() for line in report.cannot_conclude)


def test_underpowered_sample_cannot_conclude(underpowered):
    report = analyze(underpowered, "Is there a difference in y between the groups?")
    assert report.verdict == Verdict.cannot_conclude
    assert any("no effect" in line.lower() for line in report.cannot_conclude)


def test_leakage_predictor_is_dropped():
    rng = np.random.default_rng(3)
    n = 150
    age = rng.normal(50, 10, n)
    df = pd.DataFrame(
        {"patient_id": np.arange(n), "age": age, "bp": 0.5 * age + rng.normal(0, 5, n)}
    )
    report = analyze(
        df,
        "Predict bp",
        hints=PlanHints(outcome="bp", predictors=["patient_id", "age"]),
    )
    steps = report.analyses[0].revision_steps
    assert any(s.trigger == "leakage.id_predictor" for s in steps)
    assert "patient_id" not in report.analyses[0].plan.predictors


def test_clean_comparison_is_defensible():
    rng = np.random.default_rng(4)
    # Two clean normal groups with a real difference: nothing for the skeptic to fix.
    df = pd.DataFrame(
        {
            "arm": ["a"] * 50 + ["b"] * 50,
            "score": np.concatenate([rng.normal(0, 1, 50), rng.normal(1.0, 1, 50)]),
        }
    )
    report = analyze(df, "Does score differ by arm?")
    assert report.verdict == Verdict.defensible
    assert report.analyses[0].result.method == "Student's t-test"
    assert report.analyses[0].revision_steps == []


def test_unmappable_question_declines():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    report = analyze(df, "Build me a recommendation engine")
    assert report.verdict == Verdict.declined
    assert report.declined is not None
