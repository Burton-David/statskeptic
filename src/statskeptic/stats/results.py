"""The typed results every vetted routine returns.

These models are the contract the planner, critic, and report all read. They are
frozen on purpose: a statistical result you can still mutate after the fact is not a
trustworthy record, and the whole product rests on results being auditable.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

JSONScalar = str | int | float | bool | None


class Severity(str, Enum):
    """How much a violated assumption or a critique threatens the conclusion.

    Shared by `AssumptionCheck` (severity-if-violated) and `Critique`. Only `high`
    and `critical` drive the revision loop; the rest are reported but do not force a
    re-run. The ordering matters: it is what the engine sorts and compares on.
    """

    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_ORDER[self]


_SEVERITY_ORDER = {
    Severity.info: 0,
    Severity.low: 1,
    Severity.medium: 2,
    Severity.high: 3,
    Severity.critical: 4,
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AssumptionCheck(_Frozen):
    name: str
    holds: bool
    # The figure behind the decision (Shapiro W, Levene W, min expected cell count).
    # Some checks are honest assertions about the design, not tests, so both may be None.
    statistic: float | None = None
    p_value: float | None = None
    threshold: float | None = None
    # What it costs the conclusion IF this check fails. Set at the check site because
    # the same violated assumption means different things at different sample sizes.
    severity: Severity = Severity.medium
    detail: str = ""


class EffectSize(_Frozen):
    name: str
    value: float
    # Many effect sizes have no agreed closed-form CI (Cramer's V, eta-squared). We
    # leave these None and say so in the report rather than invent an interval.
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None
    # "small/medium/large" by a named convention, or None. Optional on purpose:
    # forcing a benchmark where none is agreed is exactly the over-claiming we avoid.
    interpretation: str | None = None


class ConfidenceInterval(_Frozen):
    # Naming the quantity matters: a CI on a mean difference and a CI on r are different
    # objects and a reader must not confuse them.
    quantity: str
    level: float
    low: float
    high: float
    method: str


class Computation(_Frozen):
    """The reproducibility record. A human re-runs this and gets the same numbers."""

    callable: str
    args: dict[str, JSONScalar]
    library: str
    library_version: str
    # sha256 of the serialized inputs. "Independently re-runnable" only means something
    # if a verifier can confirm they re-ran it on the same data.
    input_digest: str | None = None
    code_snippet: str | None = None


Family = Literal["two_group", "association", "k_group", "regression"]


class StatResult(_Frozen):
    method: str
    family: Family
    statistic: float
    statistic_name: str
    p_value: float
    n: int
    effect: EffectSize
    # Optional because some valid effect sizes (Cramer's V, eta-squared) have no agreed
    # closed-form interval; we report None and say so rather than invent one.
    ci: ConfidenceInterval | None = None
    assumptions: list[AssumptionCheck]
    computation: Computation
    alpha: float = 0.05
    # Set by the revision loop when a multiple-comparisons correction is applied across
    # a family of tests. None means no correction was warranted or applied.
    p_value_corrected: float | None = None
    correction: str | None = None
    # Methodological asides (ties present, normal approximation used). Never numbers,
    # never a substitute for an AssumptionCheck.
    notes: list[str] = []

    @property
    def effective_p(self) -> float:
        return (
            self.p_value if self.p_value_corrected is None else self.p_value_corrected
        )

    @property
    def is_significant(self) -> bool:
        return self.effective_p < self.alpha

    def failed_checks(self) -> list[AssumptionCheck]:
        return [c for c in self.assumptions if not c.holds]

    def worst_violation(self) -> Severity | None:
        failed = self.failed_checks()
        if not failed:
            return None
        return max((c.severity for c in failed), key=lambda s: s.rank)


class TwoGroupResult(StatResult):
    family: Family = "two_group"
    n1: int
    n2: int
    mean1: float
    mean2: float
    # Hodges-Lehmann shift for Mann-Whitney (scipy gives no location estimate of its own).
    # None for the parametric tests, where the mean difference is the location estimate.
    hodges_lehmann: float | None = None


class AssociationResult(StatResult):
    family: Family = "association"
    n_pairs: int
    # Populated only for chi-square. expected_min is what Cochran's rule reads.
    dof: int | None = None
    expected_min: float | None = None
    table_shape: tuple[int, int] | None = None


class KGroupResult(StatResult):
    family: Family = "k_group"
    group_ns: dict[str, int]
    dof_between: int
    dof_within: int | None = None


class RegressionResult(StatResult):
    family: Family = "regression"
    model_type: Literal["ols", "logistic"]
    params: dict[str, float]
    param_pvalues: dict[str, float]
    param_cis: dict[str, ConfidenceInterval]
    nobs: int
    converged: bool = True
    # OLS only.
    r_squared: float | None = None
    adj_r_squared: float | None = None
    vif: dict[str, float] | None = None
    # Logistic only. params/param_cis are reported as odds ratios for this model.
    mcfadden_pseudo_r2: float | None = None
