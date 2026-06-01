"""Characterize a dataframe into a typed `DataProfile`.

Profiling is descriptive only. It computes the shape and distribution facts the planner
and critic need, and it makes the judgment calls that decide method validity later: what
counts as a grouping factor, what looks like an identifier, how skewed a column is.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats

from .models import ColumnKind, ColumnProfile, DataProfile, NumericSummary

# Shapiro over-rejects past this; the profiler reports skew instead of a misleading p.
_SHAPIRO_MAX_N = 5000
# A numeric column of just a few integer codes is almost always a factor (arm 1/2/3,
# a Likert item), not a measurement. A genuine small-range count is the rare loser here.
_DISCRETE_NUMERIC_MAX_LEVELS = 10
_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|index|key|code)(_|$)", re.IGNORECASE)
_VALUE_COUNT_CAP = 20


def build_profile(df: pd.DataFrame) -> DataProfile:
    n_rows = len(df)
    columns = [_profile_column(df[name], name, n_rows) for name in df.columns]
    numeric_names = [c.name for c in columns if c.kind == ColumnKind.numeric]
    return DataProfile(
        n_rows=n_rows,
        n_cols=df.shape[1],
        columns=columns,
        duplicate_rows=int(df.duplicated().sum()),
        likely_id_columns=[c.name for c in columns if c.is_likely_id],
        correlations=_correlations(df, numeric_names),
    )


def _profile_column(series: pd.Series, name: str, n_rows: int) -> ColumnProfile:
    n_missing = int(series.isna().sum())
    n = int(len(series) - n_missing)
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    cardinality_fraction = n_unique / n_rows if n_rows else 0.0
    kind = _classify(series, non_null, n_unique)
    # The cardinality heuristic only applies to label-like columns. A near-unique
    # datetime is a time axis, not an identifier, and flagging it would wrongly trip the
    # leakage check when that column is the predictor in a trend.
    is_likely_id = bool(_ID_NAME.search(name)) or (
        cardinality_fraction > 0.95
        and kind in (ColumnKind.categorical, ColumnKind.text)
    )

    numeric_summary = None
    value_counts = None
    if kind == ColumnKind.numeric:
        numeric_summary = _numeric_summary(non_null)
    elif kind in (ColumnKind.categorical, ColumnKind.boolean):
        counts = non_null.value_counts().head(_VALUE_COUNT_CAP)
        value_counts = {str(k): int(v) for k, v in counts.items()}

    return ColumnProfile(
        name=name,
        kind=kind,
        n=n,
        n_missing=n_missing,
        missing_fraction=n_missing / n_rows if n_rows else 0.0,
        n_unique=n_unique,
        cardinality_fraction=cardinality_fraction,
        is_likely_id=is_likely_id,
        numeric=numeric_summary,
        value_counts=value_counts,
    )


def _classify(series: pd.Series, non_null: pd.Series, n_unique: int) -> ColumnKind:
    if n_unique <= 1:
        return ColumnKind.constant
    if pdt.is_datetime64_any_dtype(series):
        return ColumnKind.datetime
    if pdt.is_bool_dtype(series) or n_unique == 2:
        return ColumnKind.boolean
    if pdt.is_numeric_dtype(series):
        all_integer = bool((non_null == non_null.round()).all())
        if all_integer and n_unique <= _DISCRETE_NUMERIC_MAX_LEVELS:
            return ColumnKind.categorical
        return ColumnKind.numeric
    # Object/string: a high-cardinality free-text field is not a grouping factor.
    if n_unique > 50 and n_unique / max(len(non_null), 1) > 0.5:
        return ColumnKind.text
    return ColumnKind.categorical


def _numeric_summary(non_null: pd.Series) -> NumericSummary:
    arr = non_null.to_numpy(dtype=float)
    q25, median, q75 = (float(np.quantile(arr, q)) for q in (0.25, 0.5, 0.75))
    iqr = q75 - q25
    lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    n_outliers = int(((arr < lower) | (arr > upper)).sum())

    normal: bool | None = None
    normality_p: float | None = None
    if 3 <= arr.size <= _SHAPIRO_MAX_N and np.ptp(arr) > 0.0:
        _, p = stats.shapiro(arr)
        normal = bool(p >= 0.05)
        normality_p = float(p)

    return NumericSummary(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        minimum=float(arr.min()),
        q25=q25,
        median=median,
        q75=q75,
        maximum=float(arr.max()),
        skew=float(stats.skew(arr)),
        kurtosis=float(stats.kurtosis(arr)),
        n_outliers_iqr=n_outliers,
        normal=normal,
        normality_p=normality_p,
    )


def _correlations(
    df: pd.DataFrame, numeric_names: list[str]
) -> dict[str, dict[str, float]]:
    if len(numeric_names) < 2:
        return {}
    corr = df[numeric_names].corr(method="pearson")
    out: dict[str, dict[str, float]] = {}
    for a in numeric_names:
        row = {
            b: float(corr.loc[a, b])
            for b in numeric_names
            if a != b and not pd.isna(corr.loc[a, b])
        }
        if row:
            out[a] = row
    return out
