"""The signature behavior: the skeptic catches the planted error and the revision loop
fixes what it can, leaving the rest as honest caveats."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statskeptic.critique.engine import CritiqueContext, run_critique
from statskeptic.critique.models import Verdict
from statskeptic.critique.revise import (
    apply_holm,
    decide_verdict,
    multiple_comparisons_critique,
    revise,
)
from statskeptic.execution import execute
from statskeptic.plan.models import AnalysisPlan, Method, QuestionType
from statskeptic.profile import build_profile
from statskeptic.stats import pearson


def _ctx(plan, df):
    profile = build_profile(df)
    result = execute(plan, df)
    return CritiqueContext(plan, result, profile, df), profile, result


def test_normality_critique_drives_one_switch_to_mann_whitney(skewed_two_group):
    df = skewed_two_group
    plan = AnalysisPlan(
        question="q",
        question_type=QuestionType.comparison,
        method=Method.students_t,
        outcome="recovery",
        group="treatment",
    )
    profile = build_profile(df)
    result = execute(plan, df)
    outcome = revise(plan, result, profile, df)
    assert outcome.result.method == "Mann-Whitney U"
    # Exactly one switch: the loop must not oscillate back and forth.
    assert len(outcome.steps) == 1
    assert outcome.steps[0].trigger == "assumption.normality"


def test_equal_variance_critique_switches_to_welch():
    rng = np.random.default_rng(0)
    # Equal-ish shape (so normality holds) but very different spread between groups.
    a = rng.normal(0, 1, 60)
    b = rng.normal(0.2, 6, 60)
    df = pd.DataFrame({"g": ["a"] * 60 + ["b"] * 60, "y": np.concatenate([a, b])})
    plan = AnalysisPlan(
        question="q",
        question_type=QuestionType.comparison,
        method=Method.students_t,
        outcome="y",
        group="g",
    )
    profile = build_profile(df)
    outcome = revise(plan, execute(plan, df), profile, df)
    assert outcome.result.method == "Welch's t-test"


def test_confounding_critique_fires_on_causal_question(confounded):
    df = confounded
    plan = AnalysisPlan(
        question="Does exercise cause better health?",
        question_type=QuestionType.association,
        method=Method.pearson,
        outcome="exercise",
        predictors=["health"],
    )
    ctx, _, _ = _ctx(plan, df)
    crits = run_critique(ctx)
    conf = [c for c in crits if c.check_id == "confounding.causal_language"]
    assert conf and conf[0].mechanical_fix is None  # caveat, not auto-fixable
    assert "age" in conf[0].evidence  # named the candidate confounder


def test_power_critique_fires_when_underpowered(underpowered):
    df = underpowered
    plan = AnalysisPlan(
        question="Is there a difference?",
        question_type=QuestionType.comparison,
        method=Method.students_t,
        outcome="y",
        group="grp",
    )
    ctx, _, _ = _ctx(plan, df)
    crits = run_critique(ctx)
    assert any(c.check_id == "power.underpowered" for c in crits)


def test_multiple_comparisons_demotes_chance_findings():
    rng = np.random.default_rng(5)
    # 20 independent noise correlations against noise; several will look "significant".
    results = [pearson(rng.normal(0, 1, 100), rng.normal(0, 1, 100)) for _ in range(20)]
    mc = multiple_comparisons_critique(results, "outcome", 0.05)
    assert mc is not None
    corrected = apply_holm(results, 0.05)
    assert sum(r.is_significant for r in corrected) <= sum(
        r.p_value < 0.05 for r in results
    )
    # After correction the family is marked, so the critique no longer re-fires.
    assert multiple_comparisons_critique(corrected, "outcome", 0.05) is None


def test_verdict_is_cannot_conclude_with_unfixable_high():
    from statskeptic.critique.models import Critique, CritiqueCategory
    from statskeptic.stats.results import Severity

    caveat = Critique(
        check_id="confounding.causal_language",
        category=CritiqueCategory.confounding,
        severity=Severity.high,
        title="t",
        evidence="e",
        remedy="r",
    )
    assert decide_verdict([caveat]) == Verdict.cannot_conclude
    assert decide_verdict([]) == Verdict.defensible


def test_outlier_sensitivity_flips_significance():
    rng = np.random.default_rng(0)
    # No real difference between the groups; a few large values planted in b manufacture
    # a "significant" t-test (p=0.019) that collapses (p=0.060) once they are dropped.
    a = rng.normal(0, 1, 40)
    b = rng.normal(0, 1, 40)
    b[:3] = [6, 7, 8]
    df = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 40, "y": np.concatenate([a, b])})
    plan = AnalysisPlan(
        question="q",
        question_type=QuestionType.comparison,
        method=Method.students_t,
        outcome="y",
        group="g",
    )
    ctx, _, _ = _ctx(plan, df)
    crits = run_critique(ctx)
    assert any(c.check_id == "outliers.sensitivity" for c in crits)
