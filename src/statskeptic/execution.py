"""Run the plan against the data: pull the right columns, hand them to the vetted
routine the plan named, and return its result. This is the only place that knows how a
`Method` maps to a function and to columns, so the planner and the revision loop both
go through it and the result always carries an accurate computation record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from . import stats
from .errors import AnalysisError
from .plan.models import AnalysisPlan, Method
from .stats.results import StatResult


def execute(plan: AnalysisPlan, df: pd.DataFrame, alpha: float = 0.05) -> StatResult:
    if plan.method is None:
        raise AnalysisError("cannot execute a screen as a single test")
    dispatch = {
        Method.students_t: _two_group,
        Method.welch_t: _two_group,
        Method.mann_whitney_u: _two_group,
        Method.anova_oneway: _k_group,
        Method.kruskal: _k_group,
        Method.pearson: _correlation,
        Method.spearman: _correlation,
        Method.chi_square: _contingency,
        Method.fisher_exact: _contingency,
        Method.ols: _regression,
        Method.logistic: _regression,
    }
    return dispatch[plan.method](plan, df, alpha)


def _two_levels(series: pd.Series) -> list[object]:
    return sorted(series.dropna().unique().tolist(), key=str)


def _two_group(plan: AnalysisPlan, df: pd.DataFrame, alpha: float) -> StatResult:
    assert plan.outcome and plan.group and plan.method
    sub = df[[plan.outcome, plan.group]].dropna()
    levels = _two_levels(sub[plan.group])
    if len(levels) != 2:
        raise AnalysisError(
            f"a two-group test needs exactly two levels in {plan.group!r}, found {len(levels)}"
        )
    a = sub.loc[sub[plan.group] == levels[0], plan.outcome].to_numpy(dtype=float)
    b = sub.loc[sub[plan.group] == levels[1], plan.outcome].to_numpy(dtype=float)
    labels = {"label_a": str(levels[0]), "label_b": str(levels[1])}
    routines = {
        Method.students_t: stats.students_t,
        Method.welch_t: stats.welch_t,
        Method.mann_whitney_u: stats.mann_whitney_u,
    }
    return routines[plan.method](a, b, alpha=alpha, **labels)


def _k_group(plan: AnalysisPlan, df: pd.DataFrame, alpha: float) -> StatResult:
    assert plan.outcome and plan.group
    sub = df[[plan.outcome, plan.group]].dropna()
    levels = _two_levels(sub[plan.group])
    groups = [
        sub.loc[sub[plan.group] == lv, plan.outcome].to_numpy(dtype=float)
        for lv in levels
    ]
    labels = [str(lv) for lv in levels]
    routine = (
        stats.anova_oneway if plan.method == Method.anova_oneway else stats.kruskal
    )
    return routine(groups, labels, alpha=alpha)


def _correlation(plan: AnalysisPlan, df: pd.DataFrame, alpha: float) -> StatResult:
    assert plan.outcome and plan.predictors
    x = df[plan.outcome].to_numpy(dtype=float)
    y = df[plan.predictors[0]].to_numpy(dtype=float)
    routine = stats.pearson if plan.method == Method.pearson else stats.spearman
    return routine(x, y, label_x=plan.outcome, label_y=plan.predictors[0], alpha=alpha)


def _contingency(plan: AnalysisPlan, df: pd.DataFrame, alpha: float) -> StatResult:
    assert plan.outcome and plan.predictors
    sub = df[[plan.outcome, plan.predictors[0]]].dropna()
    table = pd.crosstab(sub[plan.outcome], sub[plan.predictors[0]]).to_numpy()
    if plan.method == Method.fisher_exact:
        return stats.fisher_exact(table, alpha=alpha)
    return stats.chi_square(table, alpha=alpha)


def _to_numeric_predictor(series: pd.Series) -> np.ndarray:
    if pdt.is_datetime64_any_dtype(series):
        # A trend is the slope of the outcome against elapsed time; days since the first
        # observation is the natural, interpretable unit for that slope.
        days = (series - series.min()).dt.total_seconds() / 86400.0
        as_days: np.ndarray = days.to_numpy(dtype=float)
        return as_days
    values: np.ndarray = series.to_numpy(dtype=float)
    return values


def _regression(plan: AnalysisPlan, df: pd.DataFrame, alpha: float) -> StatResult:
    assert plan.outcome and plan.predictors
    cols = [plan.outcome, *plan.predictors]
    sub = df[cols].dropna()
    exog = np.column_stack([_to_numeric_predictor(sub[p]) for p in plan.predictors])
    if plan.method == Method.logistic:
        y = _binary_outcome(sub[plan.outcome])
        return stats.logistic(y, exog, list(plan.predictors), alpha=alpha)
    y = sub[plan.outcome].to_numpy(dtype=float)
    return stats.ols(y, exog, list(plan.predictors), alpha=alpha)


def _binary_outcome(series: pd.Series) -> np.ndarray:
    if pdt.is_bool_dtype(series):
        out: np.ndarray = series.to_numpy(dtype=float)
        return out
    levels = sorted(series.dropna().unique().tolist(), key=str)
    if len(levels) != 2:
        raise AnalysisError("logistic regression needs a two-level outcome")
    mapping = {levels[0]: 0.0, levels[1]: 1.0}
    mapped: np.ndarray = series.map(mapping).to_numpy(dtype=float)
    return mapped
