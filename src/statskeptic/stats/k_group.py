"""Three or more groups: one-way ANOVA and Kruskal-Wallis.

scipy's f_oneway and kruskal return only the test statistic and p, so the effect sizes
(eta-squared, epsilon-squared) are computed here from the sums of squares. Computing
eta-squared by hand keeps it auditable and consistent with the F that f_oneway returns.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..errors import AnalysisError
from . import assumptions
from ._support import captured_warnings, computation
from .results import AssumptionCheck, EffectSize, KGroupResult, Severity


def _guard_k_group(groups: list[np.ndarray], n: int, k: int) -> None:
    if k < 2:
        raise AnalysisError(f"comparing groups needs at least 2 groups; got {k}")
    if n <= k:
        # df_within is n - k; at or below zero there is no within-group variance to test.
        raise AnalysisError(f"not enough observations ({n}) for {k} groups")
    if np.ptp(np.concatenate(groups)) == 0:
        raise AnalysisError("every value is identical across all groups")


def _clean_groups(
    groups: list[np.ndarray], labels: list[str]
) -> tuple[list[np.ndarray], list[str]]:
    out, out_labels = [], []
    for g, lab in zip(groups, labels, strict=True):
        arr = np.asarray(g, dtype=float)
        arr = arr[np.isfinite(arr)]  # drop inf alongside nan
        if arr.size > 0:
            out.append(arr)
            out_labels.append(lab)
    return out, out_labels


def _interpret_eta_squared(eta2: float) -> str:
    # Cohen 1988 for eta-squared: 0.01 / 0.06 / 0.14.
    if eta2 < 0.01:
        return "negligible (Cohen)"
    if eta2 < 0.06:
        return "small (Cohen)"
    if eta2 < 0.14:
        return "medium (Cohen)"
    return "large (Cohen)"


def anova_oneway(
    groups: list[np.ndarray],
    labels: list[str],
    *,
    alpha: float = 0.05,
) -> KGroupResult:
    groups, labels = _clean_groups(groups, labels)
    k = len(groups)
    all_values = np.concatenate(groups) if groups else np.array([])
    n = all_values.size
    _guard_k_group(groups, n, k)
    grand_mean = all_values.mean()

    ss_between = float(sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0

    with captured_warnings():
        res = stats.f_oneway(*groups)
    # A finite check is not enough: with negligible within-group variation the error term
    # is zero or rounds negative, so F comes back inf, nan, or a meaningless negative and
    # p is nan. Any of those means there is nothing for the test to weigh against.
    if (
        not (np.isfinite(res.statistic) and np.isfinite(res.pvalue))
        or res.statistic < 0
    ):
        raise AnalysisError(
            "no usable within-group variation; the F-ratio is undefined"
        )

    normality_checks = [
        assumptions.check_normality(g, lab, alpha)
        for g, lab in zip(groups, labels, strict=True)
    ]
    return KGroupResult(
        method="one-way ANOVA",
        statistic=float(res.statistic),
        statistic_name="F",
        p_value=float(res.pvalue),
        n=n,
        group_ns={lab: int(g.size) for g, lab in zip(groups, labels, strict=True)},
        dof_between=k - 1,
        dof_within=n - k,
        effect=EffectSize(
            name="eta_squared",
            value=eta2,
            interpretation=_interpret_eta_squared(eta2),
        ),
        ci=None,
        assumptions=[
            *normality_checks,
            # Welch ANOVA, the clean fix for unequal variance, is not in the v1 toolset,
            # so this is a caveat (medium), not a one-line method swap.
            assumptions.check_equal_variance(groups, alpha, severity=Severity.medium),
            assumptions.check_min_group_size(groups),
            assumptions.independence_note("independent groups"),
        ],
        computation=computation("scipy.stats.f_oneway", {}, "scipy", *groups),
        alpha=alpha,
        notes=["eta-squared computed from the sums of squares (SS_between / SS_total)"],
    )


def kruskal(
    groups: list[np.ndarray],
    labels: list[str],
    *,
    alpha: float = 0.05,
) -> KGroupResult:
    groups, labels = _clean_groups(groups, labels)
    k = len(groups)
    n = sum(g.size for g in groups)
    _guard_k_group(groups, n, k)
    with captured_warnings():
        res = stats.kruskal(*groups)
    h = float(res.statistic)

    # Epsilon-squared for Kruskal-Wallis (Tomczak & Tomczak 2014): (H - k + 1)/(n - k).
    eps2 = (h - k + 1) / (n - k) if n > k else 0.0

    return KGroupResult(
        method="Kruskal-Wallis",
        statistic=h,
        statistic_name="H",
        p_value=float(res.pvalue),
        n=n,
        group_ns={lab: int(g.size) for g, lab in zip(groups, labels, strict=True)},
        dof_between=k - 1,
        effect=EffectSize(
            name="epsilon_squared",
            value=float(eps2),
            interpretation=_interpret_eta_squared(float(eps2)),
        ),
        ci=None,
        assumptions=[
            assumptions.check_min_group_size(groups),
            AssumptionCheck(
                name="shape_homogeneity",
                holds=True,
                severity=Severity.info,
                detail=(
                    "Kruskal-Wallis reads as a test of equal medians only when the group "
                    "distributions share a shape; otherwise it tests stochastic dominance"
                ),
            ),
            assumptions.independence_note("independent groups"),
        ],
        computation=computation("scipy.stats.kruskal", {}, "scipy", *groups),
        alpha=alpha,
        notes=["epsilon-squared per Tomczak & Tomczak (2014)"],
    )
