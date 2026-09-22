"""Regression: OLS and binary logistic, via statsmodels.

Logistic separation is the headline trap here: when a predictor separates the classes
the maximum-likelihood estimate diverges, so the odds ratios are meaningless even when
the fitter prints finite numbers. Quasi-separation is flagged on the result; perfect
separation, where no MLE exists, is raised rather than dressed up as an answer.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from ..errors import AnalysisError
from . import assumptions
from ._support import captured_warnings, computation, messages
from .results import (
    AssumptionCheck,
    ConfidenceInterval,
    EffectSize,
    RegressionResult,
    Severity,
)


def _design(exog: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    design = sm.add_constant(np.asarray(exog, dtype=float), has_constant="add")
    return design, ["const", *feature_names]


def _guard_design(y: np.ndarray, design: np.ndarray) -> None:
    if design.shape[0] <= design.shape[1]:
        raise AnalysisError(
            f"not enough observations ({design.shape[0]}) for {design.shape[1]} terms"
        )
    if np.ptp(y) == 0:
        raise AnalysisError("the outcome is constant; there is nothing to model")
    # A rank-deficient design means a predictor is a linear combination of the others;
    # the coefficients are not identifiable and statsmodels would divide by zero.
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise AnalysisError(
            "predictors are perfectly collinear (singular design); drop a redundant one"
        )


def _conf_int_map(
    ci_array: np.ndarray, names: list[str], level: float, transform: str
) -> dict[str, ConfidenceInterval]:
    out = {}
    for i, name in enumerate(names):
        lo, hi = float(ci_array[i, 0]), float(ci_array[i, 1])
        if transform == "exp":
            # Under separation the log-odds CI explodes; exp overflows to inf, which is
            # the honest answer (an unbounded odds ratio), so silence only the warning.
            with np.errstate(over="ignore"):
                lo, hi = float(np.exp(lo)), float(np.exp(hi))
        quantity = f"odds_ratio[{name}]" if transform == "exp" else f"slope[{name}]"
        out[name] = ConfidenceInterval(
            quantity=quantity, level=level, low=lo, high=hi, method="wald"
        )
    return out


def ols(
    y: np.ndarray,
    exog: np.ndarray,
    feature_names: list[str],
    *,
    alpha: float = 0.05,
) -> RegressionResult:
    y = np.asarray(y, dtype=float)
    design, names = _design(exog, feature_names)
    _guard_design(y, design)
    with captured_warnings():
        res = sm.OLS(y, design).fit()

    params = {name: float(v) for name, v in zip(names, res.params, strict=True)}
    pvalues = {name: float(v) for name, v in zip(names, res.pvalues, strict=True)}
    param_cis = _conf_int_map(res.conf_int(alpha=alpha), names, 1 - alpha, "identity")

    vif = None
    checks: list[AssumptionCheck] = [
        assumptions.check_residual_normality(np.asarray(res.resid), alpha),
        assumptions.check_homoscedasticity(np.asarray(res.resid), design, alpha),
    ]
    if len(feature_names) >= 2:
        vif = {
            feature_names[i - 1]: float(variance_inflation_factor(design, i))
            for i in range(1, design.shape[1])
        }
        checks.append(assumptions.check_multicollinearity(vif))
    checks.append(
        AssumptionCheck(
            name="linearity",
            holds=True,
            severity=Severity.info,
            detail="OLS assumes the mean of y is linear in the predictors",
        )
    )
    checks.append(assumptions.independence_note("independent observations"))

    return RegressionResult(
        method="ordinary least squares",
        model_type="ols",
        statistic=float(res.fvalue),
        statistic_name="F",
        p_value=float(res.f_pvalue),
        n=int(res.nobs),
        nobs=int(res.nobs),
        params=params,
        param_pvalues=pvalues,
        param_cis=param_cis,
        r_squared=float(res.rsquared),
        adj_r_squared=float(res.rsquared_adj),
        vif=vif,
        effect=EffectSize(
            name="r_squared",
            value=float(res.rsquared),
            interpretation=None,
        ),
        ci=None,
        assumptions=checks,
        computation=computation(
            "statsmodels.api.OLS", {"add_constant": True}, "statsmodels"
        ),
        alpha=alpha,
    )


def logistic(
    y: np.ndarray,
    exog: np.ndarray,
    feature_names: list[str],
    *,
    alpha: float = 0.05,
) -> RegressionResult:
    y = np.asarray(y, dtype=float)
    if not np.isin(np.unique(y[~np.isnan(y)]), [0.0, 1.0]).all():
        raise AnalysisError("logistic regression needs a 0/1 outcome")
    if np.ptp(y) == 0:
        raise AnalysisError("the outcome has only one class; there is nothing to model")
    design, names = _design(exog, feature_names)
    _guard_design(y, design)

    try:
        with captured_warnings() as record:
            res = sm.Logit(y, design).fit(disp=0)
        warn_msgs = messages(record)
    except (PerfectSeparationError, LinAlgError) as exc:
        # No MLE exists: perfect separation leaves the Hessian singular (statsmodels
        # surfaces this as either a PerfectSeparationError or a LinAlgError depending on
        # the path). Reporting an odds ratio here would be a fabricated number.
        raise AnalysisError(
            "the logistic model is not identifiable (perfect separation), so the odds "
            f"ratios do not exist ({exc})"
        ) from exc

    # statsmodels 0.15 stopped raising PerfectSeparationError. It now emits
    # PerfectSeparationWarning and returns a diverged fit, so the handler above never
    # fires on current versions and perfect separation would be demoted to the
    # quasi-separation path below — reported with a caveat instead of refused. The
    # warning carries the same meaning the exception did: no MLE exists.
    if any(issubclass(w.category, PerfectSeparationWarning) for w in record):
        raise AnalysisError(
            "the logistic model is not identifiable (perfect separation), so the odds "
            "ratios do not exist (statsmodels reported perfect separation)"
        )

    # Quasi-separation rarely announces itself; it shows up as a fit that did not
    # converge or as coefficients blowing up on the log-odds scale. Treat any of those,
    # or an explicit separation warning, as a reason to distrust the odds ratios.
    mle_converged = bool(res.mle_retvals.get("converged", True))
    exploding = bool(np.any(np.abs(np.asarray(res.params, dtype=float)) > 15))
    warned = any(
        "separation" in m.lower() or "perfectly predicted" in m.lower()
        for m in warn_msgs
    )
    separated = warned or not mle_converged or exploding
    sep_detail = (
        "the fit did not converge or shows exploding coefficients, the signature of "
        "(quasi-)separation; odds ratios are unreliable"
        if separated
        else "no separation detected"
    )
    converged = mle_converged and not separated

    # statsmodels reports coefficients in log-odds; the interpretable scale is the odds
    # ratio, so params and their CIs are exponentiated here.
    with np.errstate(over="ignore"):
        odds_ratios = {
            name: float(np.exp(v)) for name, v in zip(names, res.params, strict=True)
        }
    pvalues = {name: float(v) for name, v in zip(names, res.pvalues, strict=True)}
    param_cis = _conf_int_map(res.conf_int(alpha=alpha), names, 1 - alpha, "exp")

    n_events = int((y == 1).sum())
    epv = min(n_events, int((y == 0).sum())) / max(len(feature_names), 1)
    checks = [
        assumptions.separation_check(not separated, sep_detail),
        AssumptionCheck(
            name="events_per_variable",
            holds=epv >= 10,
            statistic=float(epv),
            threshold=10.0,
            severity=Severity.medium,
            detail=(
                f"{epv:.1f} events per predictor; below ~10 (Peduzzi 1996) the "
                "coefficients are unstable"
            ),
        ),
        assumptions.independence_note("independent observations"),
    ]

    return RegressionResult(
        method="logistic regression",
        model_type="logistic",
        statistic=float(res.llr),
        statistic_name="LR chi2",
        p_value=float(res.llr_pvalue),
        n=int(res.nobs),
        nobs=int(res.nobs),
        converged=converged,
        params=odds_ratios,
        param_pvalues=pvalues,
        param_cis=param_cis,
        mcfadden_pseudo_r2=float(res.prsquared),
        effect=EffectSize(
            name="mcfadden_pseudo_r2",
            value=float(res.prsquared),
            interpretation=None,
        ),
        ci=None,
        assumptions=checks,
        computation=computation(
            "statsmodels.api.Logit", {"add_constant": True}, "statsmodels"
        ),
        alpha=alpha,
        notes=["coefficients reported as odds ratios (exp of the log-odds)"],
    )
