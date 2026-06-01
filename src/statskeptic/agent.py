"""The end-to-end loop: profile -> plan -> execute -> critique -> revise -> report.

`analyze` is the one public entry point. It owns the orchestration and the honesty
decisions: when the planner declines, when a screen needs a multiplicity correction,
and when surviving objections mean the only honest verdict is "cannot conclude".
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pandas.api import types as pdt

from .critique.engine import CritiqueContext, run_critique
from .critique.models import Critique, Verdict
from .critique.revise import (
    apply_holm,
    decide_verdict,
    multiple_comparisons_critique,
    revise,
)
from .execution import execute
from .plan.models import AnalysisPlan, Decline, Method, PlanHints, QuestionType
from .plan.planner import plan as make_plan
from .profile.models import DataProfile
from .profile.profiler import build_profile
from .report.models import Analysis, Report
from .stats.results import Severity, StatResult

_VERDICT_RANK = {
    Verdict.defensible: 0,
    Verdict.defensible_with_caveats: 1,
    Verdict.cannot_conclude: 2,
    Verdict.declined: 2,
}


def analyze(
    data: pd.DataFrame | str | Path,
    question: str,
    *,
    hints: PlanHints | None = None,
    alpha: float = 0.05,
) -> Report:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1 (exclusive); got {alpha}")
    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    df = _coerce_dates(df)
    profile = build_profile(df)
    planned = make_plan(question, profile, hints)

    if isinstance(planned, Decline):
        return Report(
            question=question,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            verdict=Verdict.declined,
            declined=planned,
        )

    if planned.question_type == QuestionType.screen:
        return _run_screen(planned, df, profile, question, alpha)
    return _run_single(planned, df, profile, question, alpha)


# Explicit formats only. Inferring dates from arbitrary strings is where pandas misreads
# categorical codes as dates and emits warnings; an exact format either matches every
# value in a column or that column is left exactly as it was.
_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d")


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for col in df.columns:
        series = df[col]
        # Only string columns are candidates (pandas 3 reads CSV text as the str dtype,
        # not object). Numeric, datetime, and boolean columns are left untouched.
        if not (pdt.is_object_dtype(series) or pdt.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        for fmt in _DATE_FORMATS:
            try:
                pd.to_datetime(non_null, format=fmt)
            except (ValueError, TypeError):
                continue
            if out is df:
                out = df.copy()
            out[col] = pd.to_datetime(series, format=fmt, errors="coerce")
            break
    return out


def _run_single(
    planned: AnalysisPlan,
    df: pd.DataFrame,
    profile: DataProfile,
    question: str,
    alpha: float,
) -> Report:
    result = execute(planned, df, alpha)
    outcome = revise(planned, result, profile, df, alpha)
    verdict = decide_verdict(outcome.residual)
    analysis = Analysis(
        plan=outcome.plan,
        result=outcome.result,
        revision_steps=outcome.steps,
        critiques=outcome.residual,
        verdict=verdict,
    )
    return Report(
        question=question,
        n_rows=profile.n_rows,
        n_cols=profile.n_cols,
        analyses=[analysis],
        verdict=verdict,
        cannot_conclude=_cannot_conclude(outcome.residual, outcome.result),
    )


def _run_screen(
    planned: AnalysisPlan,
    df: pd.DataFrame,
    profile: DataProfile,
    question: str,
    alpha: float,
) -> Report:
    assert planned.outcome
    subplans = [
        AnalysisPlan(
            question=question,
            question_type=QuestionType.association,
            method=Method.pearson,
            outcome=planned.outcome,
            predictors=[cand],
        )
        for cand in planned.candidates
    ]
    results = [execute(sp, df, alpha) for sp in subplans]
    per_critiques = [
        run_critique(CritiqueContext(sp, r, profile, df))
        for sp, r in zip(subplans, results, strict=True)
    ]

    mc = multiple_comparisons_critique(results, planned.outcome, alpha)
    session: list[Critique] = []
    if mc is not None:
        results = apply_holm(results, alpha)
        session = [mc]

    analyses = [
        Analysis(plan=sp, result=r, critiques=crits, verdict=decide_verdict(crits))
        for sp, r, crits in zip(subplans, results, per_critiques, strict=True)
    ]
    verdict = _worst(a.verdict for a in analyses)
    if mc is not None:
        verdict = _worse(verdict, Verdict.defensible_with_caveats)

    cannot = [line for crits in per_critiques for line in _cannot_conclude(crits, None)]
    return Report(
        question=question,
        n_rows=profile.n_rows,
        n_cols=profile.n_cols,
        analyses=analyses,
        session_critiques=session,
        verdict=verdict,
        cannot_conclude=cannot,
    )


def _worse(a: Verdict, b: Verdict) -> Verdict:
    return a if _VERDICT_RANK[a] >= _VERDICT_RANK[b] else b


def _worst(verdicts: Iterable[Verdict]) -> Verdict:
    worst = Verdict.defensible
    for v in verdicts:
        worst = _worse(worst, v)
    return worst


_CANNOT_TEMPLATES = {
    "confounding.causal_language": "Cannot conclude causation: {evidence} {remedy}",
    "power.underpowered": "Cannot conclude there is no effect: {evidence}",
}


def _cannot_conclude(residual: list[Critique], result: StatResult | None) -> list[str]:
    lines = []
    for c in residual:
        if c.severity not in (Severity.high, Severity.critical):
            continue
        template = _CANNOT_TEMPLATES.get(c.check_id)
        if template:
            lines.append(template.format(evidence=c.evidence, remedy=c.remedy))
        else:
            lines.append(
                f"Cannot rely on this result yet: {c.title.lower()} ({c.evidence})"
            )
    return lines
