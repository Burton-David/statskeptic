"""Two independent groups: Student's t, Welch's t, Mann-Whitney U.

scipy hands us the t-test mean-difference CI (>=1.11) but nothing for Mann-Whitney, so
the Hodges-Lehmann location shift and its distribution-free CI are built here by hand.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..errors import AnalysisError
from . import assumptions
from ._support import captured_warnings, computation
from .results import (
    AssumptionCheck,
    ConfidenceInterval,
    EffectSize,
    TwoGroupResult,
)


def _clean(x: np.ndarray) -> np.ndarray:
    # Drop inf as well as nan: an infinity (often an upstream divide-by-zero) is not a
    # measurement, and inf - inf in the rank estimator would otherwise produce nan.
    arr = np.asarray(x, dtype=float)
    cleaned: np.ndarray = arr[np.isfinite(arr)]
    return cleaned


def _interpret_cohens_d(d: float) -> str:
    # Cohen 1988 rules of thumb. A convention, not a law, so the report says "by Cohen".
    ad = abs(d)
    if ad < 0.2:
        return "negligible (Cohen)"
    if ad < 0.5:
        return "small (Cohen)"
    if ad < 0.8:
        return "medium (Cohen)"
    return "large (Cohen)"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n1, n2 = a.size, b.size
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = float((a.mean() - b.mean()) / sp)
    # A subnormal pooled SD (groups that are constant bar a denormalized wiggle) makes d
    # astronomical, and d**2 below would overflow float64. That regime is degenerate, not
    # a real effect, so refuse it rather than report a meaningless number.
    if not np.isfinite(d) or abs(d) > 1e150:
        raise AnalysisError(
            "pooled variance is negligible; the effect size is not computable"
        )
    # Hedges & Olkin (1985) large-sample variance of d; back it out to a normal-approx CI.
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    return d, float(se)


def _t_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    equal_var: bool,
    label_a: str,
    label_b: str,
    alpha: float,
) -> TwoGroupResult:
    a, b = _clean(a), _clean(b)
    n1, n2 = a.size, b.size
    # A t-test needs the within-group variance, so each group needs at least two points,
    # and at least one group has to vary. Without that the statistic is nan, not small.
    if n1 < 2 or n2 < 2:
        raise AnalysisError(
            f"a t-test needs at least 2 observations per group; got {n1} and {n2}"
        )
    # Guard on the pooled variance against the smallest normal float, not against zero.
    # At extreme magnitudes the spread can be real yet subnormal, which makes the pooled
    # SD underflow (Cohen's d divides by zero) or stay tiny (scipy returns t = -inf).
    # Either way the groups are effectively constant and the test is degenerate.
    pooled_var = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (
        n1 + n2 - 2
    )
    if pooled_var <= np.finfo(float).tiny:
        raise AnalysisError("both groups are effectively constant; no usable variance")
    with captured_warnings():
        res = stats.ttest_ind(a, b, equal_var=equal_var)
        ci = res.confidence_interval(confidence_level=1 - alpha)

    d, d_se = _cohens_d(a, b)
    z = stats.norm.ppf(1 - alpha / 2)

    checks: list[AssumptionCheck] = [
        assumptions.check_normality(a, label_a, alpha),
        assumptions.check_normality(b, label_b, alpha),
        assumptions.independence_note("two independent groups"),
    ]
    notes: list[str] = []
    method: str
    if equal_var:
        checks.insert(2, assumptions.check_equal_variance([a, b], alpha))
        method = "Student's t-test"
    else:
        method = "Welch's t-test"
        notes.append(
            "Welch does not assume equal variance; pooled-SD Cohen's d is an "
            "approximation here"
        )

    return TwoGroupResult(
        method=method,
        statistic=float(res.statistic),
        statistic_name="t",
        p_value=float(res.pvalue),
        n=n1 + n2,
        n1=n1,
        n2=n2,
        mean1=float(a.mean()),
        mean2=float(b.mean()),
        effect=EffectSize(
            name="cohens_d",
            value=d,
            ci_low=d - z * d_se,
            ci_high=d + z * d_se,
            ci_level=1 - alpha,
            interpretation=_interpret_cohens_d(d),
        ),
        ci=ConfidenceInterval(
            quantity=f"mean({label_a}) - mean({label_b})",
            level=1 - alpha,
            low=float(ci.low),
            high=float(ci.high),
            method="t_distribution",
        ),
        assumptions=checks,
        computation=computation(
            "scipy.stats.ttest_ind",
            {"equal_var": equal_var, "alternative": "two-sided"},
            "scipy",
            a,
            b,
        ),
        alpha=alpha,
        notes=notes,
    )


def students_t(
    a: np.ndarray,
    b: np.ndarray,
    *,
    label_a: str = "group1",
    label_b: str = "group2",
    alpha: float = 0.05,
) -> TwoGroupResult:
    return _t_test(a, b, equal_var=True, label_a=label_a, label_b=label_b, alpha=alpha)


def welch_t(
    a: np.ndarray,
    b: np.ndarray,
    *,
    label_a: str = "group1",
    label_b: str = "group2",
    alpha: float = 0.05,
) -> TwoGroupResult:
    return _t_test(a, b, equal_var=False, label_a=label_a, label_b=label_b, alpha=alpha)


def _hodges_lehmann(
    a: np.ndarray, b: np.ndarray, alpha: float
) -> tuple[float, float, float]:
    """Median of the pairwise differences a_i - b_j, with the distribution-free CI from
    Hollander and Wolfe, Nonparametric Statistical Methods (normal approximation to the
    rank count). Returns (shift, ci_low, ci_high)."""
    diffs = np.subtract.outer(a, b).ravel()
    diffs.sort()
    m = diffs.size
    shift = float(np.median(diffs))
    z = stats.norm.ppf(1 - alpha / 2)
    # C is the rank offset from each tail; the bounds sit symmetrically around the median.
    c = int(np.floor(m / 2 - z * np.sqrt(a.size * b.size * (a.size + b.size + 1) / 12)))
    c = max(c, 0)
    return shift, float(diffs[c]), float(diffs[m - 1 - c])


def mann_whitney_u(
    a: np.ndarray,
    b: np.ndarray,
    *,
    label_a: str = "group1",
    label_b: str = "group2",
    alpha: float = 0.05,
) -> TwoGroupResult:
    a, b = _clean(a), _clean(b)
    n1, n2 = a.size, b.size
    if n1 < 1 or n2 < 1:
        raise AnalysisError(
            f"a rank test needs at least 1 observation per group; got {n1} and {n2}"
        )
    if np.ptp(np.concatenate([a, b])) == 0:
        raise AnalysisError("every value is identical; there is nothing to rank")
    with captured_warnings():
        res = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")

    # Rank-biserial r = 2 U1/(n1 n2) - 1: +1 when group a always exceeds group b.
    r_rb = float(2 * res.statistic / (n1 * n2) - 1)
    shift, lo, hi = _hodges_lehmann(a, b, alpha)

    notes = [
        "Mann-Whitney compares whole distributions; read as equal medians only under a "
        "location-shift model"
    ]
    combined = np.concatenate([a, b])
    if np.unique(combined).size < combined.size:
        notes.append(
            "ties present; normal approximation with continuity correction used"
        )

    return TwoGroupResult(
        method="Mann-Whitney U",
        statistic=float(res.statistic),
        statistic_name="U",
        p_value=float(res.pvalue),
        n=n1 + n2,
        n1=n1,
        n2=n2,
        mean1=float(a.mean()),
        mean2=float(b.mean()),
        hodges_lehmann=shift,
        effect=EffectSize(name="rank_biserial_r", value=r_rb),
        ci=ConfidenceInterval(
            quantity=f"location shift ({label_a} - {label_b})",
            level=1 - alpha,
            low=lo,
            high=hi,
            method="hodges_lehmann",
        ),
        assumptions=[
            assumptions.check_min_group_size([a, b]),
            assumptions.independence_note("two independent groups"),
        ],
        computation=computation(
            "scipy.stats.mannwhitneyu",
            {"alternative": "two-sided", "method": "auto"},
            "scipy",
            a,
            b,
        ),
        alpha=alpha,
        notes=notes,
    )
