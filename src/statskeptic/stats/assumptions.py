"""Assumption checks. Each returns an `AssumptionCheck` the routine attaches to its
result and the critic later reads. The gotcha each one encodes is the whole point:
these are the lines that separate a careful analyst from a tool that runs the test and
hopes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._support import captured_warnings
from .results import AssumptionCheck, Severity

# Above this, Shapiro-Wilk over-rejects: it flags deviations too small to matter and
# would trigger a needless switch to a rank test. Past it we judge shape by skew.
_SHAPIRO_MAX_N = 5000

# Skew past this is a genuine departure from normality, not a quirk, so the mean-based
# test is the wrong tool regardless of a moderate sample size.
_SKEW_HEAVY = 1.0


def check_normality(x: np.ndarray, name: str, alpha: float = 0.05) -> AssumptionCheck:
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 3 or np.ptp(arr) == 0.0:
        # Shapiro needs n>=3 and non-constant input; below that normality is not a
        # question we can answer, so we do not pretend to.
        return AssumptionCheck(
            name=f"normality[{name}]",
            holds=True,
            severity=Severity.info,
            detail=f"normality not assessable for {name} (n={n})",
        )

    skew = float(stats.skew(arr))
    severity = _normality_severity(n, skew)

    if n > _SHAPIRO_MAX_N:
        holds = abs(skew) < _SKEW_HEAVY
        return AssumptionCheck(
            name=f"normality[{name}]",
            holds=holds,
            statistic=None,
            p_value=None,
            severity=severity,
            detail=(
                f"{name}: n={n} too large for Shapiro (over-rejects); "
                f"judged by skew={skew:.2f}"
            ),
        )

    with captured_warnings():
        w_stat, p = stats.shapiro(arr)
    holds = bool(p >= alpha)
    return AssumptionCheck(
        name=f"normality[{name}]",
        holds=holds,
        statistic=float(w_stat),
        p_value=float(p),
        threshold=alpha,
        severity=severity,
        detail=f"{name}: Shapiro W={float(w_stat):.3f}, p={float(p):.3g}, skew={skew:.2f}",
    )


def _normality_severity(n: int, skew: float) -> Severity:
    # Heavy skew keeps it serious even at moderate n; otherwise the CLT carries the
    # mean once the sample is large enough, so a small deviation matters less and less.
    if abs(skew) >= _SKEW_HEAVY:
        return Severity.high
    if n < 40:
        return Severity.medium
    return Severity.low


def check_equal_variance(
    groups: list[np.ndarray],
    alpha: float = 0.05,
    severity: Severity = Severity.high,
) -> AssumptionCheck:
    # severity defaults to high for Student's t, where Welch is a clean drop-in fix. For
    # ANOVA the honest remedy (Welch ANOVA) is not in the v1 toolset, so callers there
    # pass medium: it is a caveat, not a one-line method swap.
    cleaned = [
        np.asarray(g, dtype=float)[~np.isnan(np.asarray(g, dtype=float))]
        for g in groups
    ]
    usable = [g for g in cleaned if g.size >= 2 and np.ptp(g) > 0.0]
    if len(usable) < 2:
        return AssumptionCheck(
            name="equal_variance",
            holds=True,
            severity=Severity.info,
            detail="equal-variance test needs >=2 groups with spread",
        )

    # Levene with center='median' (Brown-Forsythe) is robust to non-normality. Bartlett
    # is not: it rejects for skew, so it would confound a variance test with a normality
    # test and is only safe once normality already holds.
    with captured_warnings():
        w_stat, p = stats.levene(*usable, center="median")
    sds = [round(float(np.std(g, ddof=1)), 3) for g in usable]
    holds = bool(p >= alpha)
    return AssumptionCheck(
        name="equal_variance",
        holds=holds,
        statistic=float(w_stat),
        p_value=float(p),
        threshold=alpha,
        severity=severity,
        detail=f"Levene (median-centered) p={float(p):.3g}; group SDs {sds}",
    )


def check_expected_cell_counts(
    expected: np.ndarray, min_expected: float = 5.0
) -> AssumptionCheck:
    exp = np.asarray(expected, dtype=float)
    min_exp = float(exp.min())
    frac_below = float((exp < min_expected).mean())
    # Cochran's rule, relaxed form: no expected count below 1 and at most 20% below 5.
    # Below that the chi-square approximation to the sampling distribution is unreliable.
    holds = min_exp >= 1.0 and frac_below <= 0.2
    return AssumptionCheck(
        name="expected_cell_counts",
        holds=holds,
        statistic=min_exp,
        threshold=min_expected,
        severity=Severity.high,
        detail=(
            f"min expected count {min_exp:.2f}; "
            f"{frac_below:.0%} of cells below {min_expected:g}"
        ),
    )


def check_min_group_size(groups: list[np.ndarray], floor: int = 5) -> AssumptionCheck:
    sizes = [
        int(np.asarray(g)[~np.isnan(np.asarray(g, dtype=float))].size) for g in groups
    ]
    smallest = min(sizes) if sizes else 0
    holds = smallest >= floor
    return AssumptionCheck(
        name="min_group_size",
        holds=holds,
        statistic=float(smallest),
        threshold=float(floor),
        # The large-sample approximations (Mann-Whitney normal approx, Kruskal chi-square)
        # lean on a handful per group; below that the p-value is only approximate.
        severity=Severity.medium,
        detail=f"smallest group n={smallest}; group sizes {sizes}",
    )


def independence_note(context: str) -> AssumptionCheck:
    """Independence is a property of the design, not the values, so we state it rather
    than fake a test for it. Saying so plainly is itself a feature."""
    return AssumptionCheck(
        name="independence",
        holds=True,
        severity=Severity.info,
        detail=(
            f"independence assumed ({context}); not verifiable from the data alone, "
            "so check the design for repeated measures, clustering, or time order"
        ),
    )


def check_linearity(x: np.ndarray, y: np.ndarray) -> AssumptionCheck:
    """Pearson measures linear association only. A large gap between the rank
    correlation and Pearson r is the tell that the relationship is monotonic but
    curved, where Pearson understates it and Spearman is the better summary."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[mask], ya[mask]
    if xa.size < 4 or np.ptp(xa) == 0.0 or np.ptp(ya) == 0.0:
        return AssumptionCheck(
            name="linearity",
            holds=True,
            severity=Severity.info,
            detail="linearity not assessable",
        )
    with captured_warnings():
        r = float(stats.pearsonr(xa, ya).statistic)
        rho = float(stats.spearmanr(xa, ya).statistic)
    gap = abs(rho) - abs(r)
    holds = gap <= 0.1
    return AssumptionCheck(
        name="linearity",
        holds=holds,
        statistic=gap,
        threshold=0.1,
        severity=Severity.medium,
        detail=f"Pearson r={r:.3f} vs Spearman rho={rho:.3f}; gap {gap:.3f}",
    )


def check_residual_normality(resid: np.ndarray, alpha: float = 0.05) -> AssumptionCheck:
    # The normality assumption in OLS is on the residuals, not on y. Testing y is the
    # common mistake; we test resid.
    check = check_normality(resid, "residuals", alpha)
    return check.model_copy(update={"name": "residual_normality"})


def check_homoscedasticity(
    resid: np.ndarray, exog: np.ndarray, alpha: float = 0.05
) -> AssumptionCheck:
    # Breusch-Pagan. Heteroscedasticity does not bias OLS coefficients but it invalidates
    # the standard errors, so every p-value and CI built on them is wrong.
    from statsmodels.stats.diagnostic import het_breuschpagan

    with captured_warnings():
        lm_stat, lm_p, _, _ = het_breuschpagan(resid, exog)
    holds = bool(lm_p >= alpha)
    return AssumptionCheck(
        name="homoscedasticity",
        holds=holds,
        statistic=float(lm_stat),
        p_value=float(lm_p),
        threshold=alpha,
        severity=Severity.high,
        detail=f"Breusch-Pagan LM p={float(lm_p):.3g}",
    )


def check_multicollinearity(vif: dict[str, float]) -> AssumptionCheck:
    # VIF > 5 inflates a coefficient's SE; > 10 is severe. Collinearity does not hurt
    # prediction but it makes individual coefficients uninterpretable.
    if not vif:
        return AssumptionCheck(
            name="multicollinearity",
            holds=True,
            severity=Severity.info,
            detail="single predictor; VIF not applicable",
        )
    worst_name = max(vif, key=lambda k: vif[k])
    worst = vif[worst_name]
    holds = worst < 10.0
    return AssumptionCheck(
        name="multicollinearity",
        holds=holds,
        statistic=worst,
        threshold=10.0,
        severity=Severity.medium,
        detail=f"max VIF {worst:.2f} on {worst_name!r}",
    )


def separation_check(holds: bool, detail: str) -> AssumptionCheck:
    # Perfect or quasi-separation: the logistic MLE diverges, so the odds ratios and
    # their CIs are meaningless even when statsmodels prints finite-looking numbers.
    return AssumptionCheck(
        name="separation",
        holds=holds,
        severity=Severity.high,
        detail=detail,
    )
