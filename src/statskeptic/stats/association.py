"""Association: Pearson r, Spearman rho, chi-square test of independence.

scipy gives a Fisher-z CI for Pearson r (>=1.11) but not for Spearman, and there is no
agreed closed-form CI for Cramer's V, so those are built or left None honestly here.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..errors import AnalysisError
from . import assumptions
from ._support import captured_warnings, computation
from .results import (
    AssociationResult,
    AssumptionCheck,
    ConfidenceInterval,
    EffectSize,
)


def _paired_clean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    # Keep only pairs where both values are finite; inf is treated as missing.
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def _guard_correlation(x: np.ndarray, y: np.ndarray) -> None:
    # A correlation needs at least three paired points and variation in both variables;
    # a constant column has no correlation to estimate, and scipy would warn or divide
    # by zero rather than say so.
    if x.size < 3:
        raise AnalysisError(
            f"a correlation needs at least 3 paired values; got {x.size}"
        )
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise AnalysisError(
            "one of the variables is constant; correlation is undefined"
        )


def _interpret_r(r: float) -> str:
    # Cohen 1988 for correlations: 0.1 / 0.3 / 0.5.
    ar = abs(r)
    if ar < 0.1:
        return "negligible (Cohen)"
    if ar < 0.3:
        return "small (Cohen)"
    if ar < 0.5:
        return "medium (Cohen)"
    return "large (Cohen)"


def pearson(
    x: np.ndarray,
    y: np.ndarray,
    *,
    label_x: str = "x",
    label_y: str = "y",
    alpha: float = 0.05,
) -> AssociationResult:
    x, y = _paired_clean(x, y)
    _guard_correlation(x, y)
    n = x.size
    with captured_warnings():
        res = stats.pearsonr(x, y)
        ci = res.confidence_interval(confidence_level=1 - alpha)
    r = float(res.statistic)
    return AssociationResult(
        method="Pearson correlation",
        statistic=r,
        statistic_name="r",
        p_value=float(res.pvalue),
        n=n,
        n_pairs=n,
        effect=EffectSize(
            name="pearson_r",
            value=r,
            ci_low=float(ci.low),
            ci_high=float(ci.high),
            ci_level=1 - alpha,
            interpretation=_interpret_r(r),
        ),
        ci=ConfidenceInterval(
            quantity="pearson_r",
            level=1 - alpha,
            low=float(ci.low),
            high=float(ci.high),
            method="fisher_z",
        ),
        assumptions=[
            assumptions.check_normality(x, label_x, alpha),
            assumptions.check_normality(y, label_y, alpha),
            assumptions.check_linearity(x, y),
            assumptions.independence_note("paired observations"),
        ],
        computation=computation(
            "scipy.stats.pearsonr", {"alternative": "two-sided"}, "scipy", x, y
        ),
        alpha=alpha,
    )


def spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    label_x: str = "x",
    label_y: str = "y",
    alpha: float = 0.05,
) -> AssociationResult:
    x, y = _paired_clean(x, y)
    _guard_correlation(x, y)
    n = x.size
    with captured_warnings():
        res = stats.spearmanr(x, y)
    rho = float(res.statistic)

    # No Spearman CI from scipy. Fisher z with the Bonett & Wright (2000) standard error
    # 1.03/sqrt(n-3), which is the Pearson SE inflated for the rank transform. Skip it
    # near |rho|=1, where atanh blows up and any interval is a degenerate [1, 1].
    lo = hi = None
    if n > 3 and abs(rho) < 0.9999:
        z = np.arctanh(rho)
        se = 1.03 / np.sqrt(n - 3)
        zcrit = stats.norm.ppf(1 - alpha / 2)
        lo = float(np.tanh(z - zcrit * se))
        hi = float(np.tanh(z + zcrit * se))

    return AssociationResult(
        method="Spearman rank correlation",
        statistic=rho,
        statistic_name="rho",
        p_value=float(res.pvalue),
        n=n,
        n_pairs=n,
        effect=EffectSize(
            name="spearman_rho",
            value=rho,
            ci_low=lo,
            ci_high=hi,
            ci_level=1 - alpha if lo is not None else None,
            interpretation=_interpret_r(rho),
        ),
        ci=(
            ConfidenceInterval(
                quantity="spearman_rho",
                level=1 - alpha,
                low=lo,
                high=hi,
                method="fisher_z_bonett_wright",
            )
            if lo is not None
            else None
        ),
        assumptions=[
            assumptions.independence_note("paired observations"),
            AssumptionCheck(
                name="monotonicity",
                holds=True,
                detail="Spearman assumes a monotonic relationship, not a linear one",
            ),
        ],
        computation=computation(
            "scipy.stats.spearmanr", {"alternative": "two-sided"}, "scipy", x, y
        ),
        alpha=alpha,
    )


def fisher_exact(
    table: np.ndarray,
    *,
    alpha: float = 0.05,
) -> AssociationResult:
    """Fisher's exact test for a 2x2 table. The honest choice over chi-square when
    expected counts are small, because it does not lean on the chi-square approximation
    that low counts break."""
    obs = np.asarray(table, dtype=float)
    if obs.shape != (2, 2):
        raise ValueError("Fisher's exact is defined here for 2x2 tables only")
    with captured_warnings():
        res = stats.fisher_exact(obs, alternative="two-sided")
    return AssociationResult(
        method="Fisher's exact test",
        statistic=float(res.statistic),
        statistic_name="odds_ratio",
        p_value=float(res.pvalue),
        n=int(obs.sum()),
        n_pairs=int(obs.sum()),
        dof=1,
        table_shape=(2, 2),
        effect=EffectSize(name="odds_ratio", value=float(res.statistic)),
        ci=None,
        assumptions=[assumptions.independence_note("each observation counted once")],
        computation=computation(
            "scipy.stats.fisher_exact", {"alternative": "two-sided"}, "scipy", obs
        ),
        alpha=alpha,
        notes=["exact test; no large-sample approximation, so small counts are fine"],
    )


def chi_square(
    table: np.ndarray,
    *,
    alpha: float = 0.05,
) -> AssociationResult:
    """Chi-square test of independence on a contingency table of counts."""
    obs = np.asarray(table, dtype=float)
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2:
        raise AnalysisError("chi-square needs a table of at least 2 rows and 2 columns")
    if (obs.sum(axis=0) == 0).any() or (obs.sum(axis=1) == 0).any():
        # An empty row or column means a category never appears; scipy cannot form the
        # expected counts and the test is meaningless for it.
        raise AnalysisError("a row or column of the table is all zeros; cannot test it")
    with captured_warnings():
        res = stats.chi2_contingency(obs, correction=True)
    chi2 = float(res.statistic)
    n = int(obs.sum())
    rows, cols = obs.shape

    # Cramer's V normalizes chi-square to [0, 1] by n and the smaller table dimension.
    v = float(np.sqrt(chi2 / (n * min(rows - 1, cols - 1)))) if n > 0 else 0.0

    notes = ["no agreed closed-form CI for Cramer's V; reported as a point estimate"]
    if rows == 2 and cols == 2:
        notes.append("2x2 table; Yates continuity correction applied")

    return AssociationResult(
        method="chi-square test of independence",
        statistic=chi2,
        statistic_name="chi2",
        p_value=float(res.pvalue),
        n=n,
        n_pairs=n,
        dof=int(res.dof),
        expected_min=float(res.expected_freq.min()),
        table_shape=(rows, cols),
        effect=EffectSize(name="cramers_v", value=v, interpretation=None),
        ci=None,
        assumptions=[
            assumptions.check_expected_cell_counts(res.expected_freq),
            assumptions.independence_note("each observation counted once"),
        ],
        computation=computation(
            "scipy.stats.chi2_contingency", {"correction": True}, "scipy", obs
        ),
        alpha=alpha,
        notes=notes,
    )
