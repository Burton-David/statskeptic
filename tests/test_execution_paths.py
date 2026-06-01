"""End-to-end coverage for every method's execution bridge through analyze(), not just
the routines in isolation. These paths (k-group, contingency, trend, missing data) were
exercised only at the unit level before; here they run the whole loop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statskeptic import Verdict, analyze


def test_anova_runs_end_to_end():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "site": ["x", "y", "z"] * 40,
            "score": np.concatenate(
                [rng.normal(0, 1, 40), rng.normal(0.4, 1, 40), rng.normal(0.8, 1, 40)]
            ),
        }
    )
    r = analyze(df, "Does score differ across site?")
    assert r.analyses[0].result.method == "one-way ANOVA"
    assert r.analyses[0].result.statistic > 0


def test_anova_switches_to_kruskal_on_skew():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "site": ["x", "y", "z"] * 40,
            "score": np.concatenate(
                [
                    rng.lognormal(0, 1, 40),
                    rng.lognormal(0.3, 1, 40),
                    rng.lognormal(0.6, 1, 40),
                ]
            ),
        }
    )
    r = analyze(df, "Does score differ across site?")
    assert r.analyses[0].result.method == "Kruskal-Wallis"
    assert any(
        s.trigger == "assumption.normality" for s in r.analyses[0].revision_steps
    )


def test_chi_square_runs_end_to_end():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "smoker": rng.choice(["yes", "no"], 200),
            "diagnosis": rng.choice(["pos", "neg"], 200),
        }
    )
    r = analyze(df, "Is smoker related to diagnosis?")
    assert r.analyses[0].result.method == "chi-square test of independence"


def test_chi_square_switches_to_fisher_on_sparse_2x2():
    # Margins force an expected count below 5, so the chi-square approximation is
    # invalid and the loop must move to Fisher's exact.
    df = pd.DataFrame(
        {
            "exposed": ["yes"] * 10 + ["no"] * 30,
            "disease": (["pos"] * 1 + ["neg"] * 9) + (["pos"] * 1 + ["neg"] * 29),
        }
    )
    r = analyze(df, "Is exposed related to disease?")
    assert r.analyses[0].result.method == "Fisher's exact test"
    assert any(
        s.trigger == "assumption.expected_counts" for s in r.analyses[0].revision_steps
    )


def test_trend_does_not_misflag_the_time_axis_as_leakage():
    # Regression for a real bug: a near-unique datetime predictor was wrongly flagged as
    # an identifier, forcing cannot_conclude on every trend.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "day": pd.to_datetime("2026-01-01") + pd.to_timedelta(range(60), unit="D"),
            "revenue": np.linspace(100, 200, 60) + rng.normal(0, 5, 60),
        }
    )
    r = analyze(df, "Is there a trend in revenue over time?")
    assert r.analyses[0].result.method == "ordinary least squares"
    assert not any(
        c.check_id == "leakage.id_predictor" for c in r.analyses[0].critiques
    )
    # A clear upward trend should be detected, not buried under a false leakage flag.
    assert r.analyses[0].result.params["day"] > 1.0


def test_missing_data_becomes_a_caveat():
    rng = np.random.default_rng(3)
    score = np.concatenate([rng.normal(0, 1, 60), rng.normal(0.8, 1, 60)])
    score[:15] = np.nan  # 12.5% missing, above the 5% threshold
    df = pd.DataFrame({"arm": ["a"] * 60 + ["b"] * 60, "score": score})
    r = analyze(df, "Does score differ by arm?")
    assert any(c.check_id == "missing.handling" for c in r.analyses[0].critiques)
    assert r.verdict == Verdict.defensible_with_caveats
