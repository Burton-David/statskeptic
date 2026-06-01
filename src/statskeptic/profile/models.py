"""The typed `DataProfile`: what the planner and critics read instead of touching the
raw frame. Descriptive statistics belong here; inferential statistics belong in stats/.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ColumnKind(str, Enum):
    numeric = "numeric"
    categorical = "categorical"
    boolean = "boolean"
    datetime = "datetime"
    text = "text"
    constant = "constant"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NumericSummary(_Frozen):
    mean: float
    std: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float
    skew: float
    kurtosis: float
    # Count beyond Tukey's 1.5*IQR fences. A count, not a verdict: outliers are not
    # errors, they are points the conclusion should not secretly hinge on.
    n_outliers_iqr: int
    # Shapiro decision, or None when n is outside the range where Shapiro is meaningful.
    normal: bool | None = None
    normality_p: float | None = None


class ColumnProfile(_Frozen):
    name: str
    kind: ColumnKind
    n: int
    n_missing: int
    missing_fraction: float
    n_unique: int
    cardinality_fraction: float
    is_likely_id: bool = False
    numeric: NumericSummary | None = None
    # For categorical/boolean: the levels and their counts (capped), so a planner can see
    # how many groups there are and how balanced they are.
    value_counts: dict[str, int] | None = None


class DataProfile(_Frozen):
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    duplicate_rows: int
    likely_id_columns: list[str]
    # Pairwise Pearson correlations among numeric columns; the confounding check reads
    # these to name a candidate third variable.
    correlations: dict[str, dict[str, float]]

    def by_name(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def numeric(self) -> list[ColumnProfile]:
        return [c for c in self.columns if c.kind == ColumnKind.numeric]

    def categorical(self) -> list[ColumnProfile]:
        # Booleans group like categoricals for the planner's purposes.
        return [
            c
            for c in self.columns
            if c.kind in (ColumnKind.categorical, ColumnKind.boolean)
        ]

    def correlation(self, a: str, b: str) -> float | None:
        return self.correlations.get(a, {}).get(b)
