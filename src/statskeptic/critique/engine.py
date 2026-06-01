"""The deterministic skeptic. Each check is a small function that reads a completed
analysis and the data behind it and returns objections grounded in the actual numbers.

Checks read the statistics the stats layer already computed; they do not re-derive a
test statistic from scratch (the one exception, the outlier refit, re-runs the same
vetted routine, so even that number originates in stats/). The registry plus the
severity-override hook is the seam a clinical or financial pack extends without editing
this file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

from .. import stats as stats_pkg
from ..plan.models import AnalysisPlan, Method
from ..profile.models import DataProfile
from ..stats.results import Severity, StatResult
from .models import Critique, CritiqueCategory, FixAction


@dataclass(frozen=True)
class CritiqueContext:
    plan: AnalysisPlan
    result: StatResult
    profile: DataProfile
    data: pd.DataFrame


CheckFn = Callable[[CritiqueContext], list[Critique]]


@dataclass
class RegisteredCheck:
    fn: CheckFn
    category: CritiqueCategory


_REGISTRY: list[RegisteredCheck] = []
# Domain packs may bump a core check's severity by check_id without rewriting it. Empty
# in the open-source core; the hook is what makes the packs a real extension point.
_SEVERITY_OVERRIDES: dict[str, Severity] = {}


def register_check(category: CritiqueCategory) -> Callable[[CheckFn], CheckFn]:
    def decorator(fn: CheckFn) -> CheckFn:
        _REGISTRY.append(RegisteredCheck(fn=fn, category=category))
        return fn

    return decorator


def run_critique(ctx: CritiqueContext) -> list[Critique]:
    found: list[Critique] = []
    for check in _REGISTRY:
        for c in check.fn(ctx):
            override = _SEVERITY_OVERRIDES.get(c.check_id)
            found.append(
                c if override is None else c.model_copy(update={"severity": override})
            )
    found.sort(key=lambda c: c.severity.rank, reverse=True)
    return found


_RANK_ANALOGUE = {
    Method.students_t: Method.mann_whitney_u,
    Method.welch_t: Method.mann_whitney_u,
    Method.anova_oneway: Method.kruskal,
    Method.pearson: Method.spearman,
}

_CAUSAL_CUES = (
    "cause",
    "causes",
    "caused",
    "causing",
    "effect of",
    "impact of",
    "due to",
    "leads to",
    "results in",
    "because of",
    "drives",
)


@register_check(CritiqueCategory.assumption)
def normality(ctx: CritiqueContext) -> list[Critique]:
    analogue = _RANK_ANALOGUE.get(ctx.plan.method) if ctx.plan.method else None
    if analogue is None:
        return []
    bad = [
        c
        for c in ctx.result.assumptions
        if c.name.startswith("normality")
        and not c.holds
        and c.severity == Severity.high
    ]
    if not bad:
        return []
    worst = bad[0]
    return [
        Critique(
            check_id="assumption.normality",
            category=CritiqueCategory.assumption,
            severity=Severity.high,
            title="Normality assumption violated",
            evidence=f"{ctx.result.method} assumes normality, but {worst.detail}",
            evidence_data={"shapiro_w": worst.statistic, "p": worst.p_value},
            remedy=f"switch to {analogue.value}, which does not assume normality",
            mechanical_fix=FixAction(
                kind="switch_method",
                target=analogue.value,
                reason="data is non-normal; the rank-based test is valid here",
            ),
        )
    ]


@register_check(CritiqueCategory.assumption)
def equal_variance(ctx: CritiqueContext) -> list[Critique]:
    if ctx.plan.method != Method.students_t:
        return []
    bad = [
        c for c in ctx.result.assumptions if c.name == "equal_variance" and not c.holds
    ]
    if not bad:
        return []
    return [
        Critique(
            check_id="assumption.equal_variance",
            category=CritiqueCategory.assumption,
            severity=Severity.high,
            title="Equal-variance assumption violated",
            evidence=f"Student's t pools the variances, but {bad[0].detail}",
            evidence_data={"levene_p": bad[0].p_value},
            remedy="use Welch's t-test, which does not assume equal variance",
            mechanical_fix=FixAction(
                kind="switch_method",
                target=Method.welch_t.value,
                reason="unequal variances bias the pooled standard error",
            ),
        )
    ]


@register_check(CritiqueCategory.assumption)
def expected_counts(ctx: CritiqueContext) -> list[Critique]:
    if ctx.plan.method != Method.chi_square:
        return []
    bad = [
        c
        for c in ctx.result.assumptions
        if c.name == "expected_cell_counts" and not c.holds
    ]
    if not bad:
        return []
    shape = getattr(ctx.result, "table_shape", None)
    is_2x2 = shape == (2, 2)
    fix = (
        FixAction(
            kind="switch_method",
            target=Method.fisher_exact.value,
            reason="exact test avoids the chi-square approximation that small counts break",
        )
        if is_2x2
        else None
    )
    remedy = (
        "use Fisher's exact test"
        if is_2x2
        else "collapse sparse categories or collect more data; the chi-square p is unreliable"
    )
    return [
        Critique(
            check_id="assumption.expected_counts",
            category=CritiqueCategory.assumption,
            severity=Severity.high,
            title="Expected cell counts too small",
            evidence=f"chi-square needs expected counts of at least 5, but {bad[0].detail}",
            evidence_data={"min_expected": bad[0].statistic},
            remedy=remedy,
            mechanical_fix=fix,
        )
    ]


@register_check(CritiqueCategory.confounding)
def confounding(ctx: CritiqueContext) -> list[Critique]:
    q = ctx.plan.question.lower()
    if not any(cue in q for cue in _CAUSAL_CUES):
        return []
    candidate = _candidate_confounder(ctx)
    if candidate:
        name, r_out, r_exp = candidate
        extra = (
            f" {name!r} correlates with both the outcome (r={r_out:.2f}) and the "
            f"exposure (r={r_exp:.2f}), so it is a candidate confounder"
        )
        data = {"confounder": name, "r_outcome": r_out, "r_exposure": r_exp}
    else:
        extra = " an unmeasured third variable could drive both"
        data = {}
    return [
        Critique(
            check_id="confounding.causal_language",
            category=CritiqueCategory.confounding,
            severity=Severity.high,
            title="Causal claim from observational data",
            evidence=(
                "the question asks about causation, but this is observational data, so "
                "association is not causation." + extra
            ),
            evidence_data=data,
            remedy=(
                "re-scope the claim to association, or use a design that supports causal "
                "inference (randomization, or adjustment for measured confounders)"
            ),
        )
    ]


def _candidate_confounder(ctx: CritiqueContext) -> tuple[str, float, float] | None:
    outcome = ctx.plan.outcome
    exposure = ctx.plan.predictors[0] if ctx.plan.predictors else None
    if not outcome or not exposure:
        return None
    best = None
    for z, row in ctx.profile.correlations.items():
        if z in (outcome, exposure):
            continue
        r_out = row.get(outcome)
        r_exp = row.get(exposure)
        if r_out is None or r_exp is None:
            continue
        if abs(r_out) >= 0.3 and abs(r_exp) >= 0.3:
            strength = abs(r_out) + abs(r_exp)
            if best is None or strength > best[3]:
                best = (z, r_out, r_exp, strength)
    return (best[0], best[1], best[2]) if best else None


@register_check(CritiqueCategory.power)
def power(ctx: CritiqueContext) -> list[Critique]:
    # Only for the t-test family, where Cohen's d and the power calculation line up.
    if ctx.result.effect.name != "cohens_d" or ctx.result.is_significant:
        return []
    n1 = getattr(ctx.result, "n1", None)
    n2 = getattr(ctx.result, "n2", None)
    if not n1 or not n2:
        return []
    # Minimum detectable effect at this n, not the observed-power fallacy (which is just
    # a restatement of the p-value). MDE answers "what could this study have caught?".
    mde = float(
        TTestIndPower().solve_power(
            effect_size=None, nobs1=n1, alpha=ctx.result.alpha, power=0.8, ratio=n2 / n1
        )
    )
    if mde < 0.8:
        return []
    needed = float(
        TTestIndPower().solve_power(effect_size=0.5, alpha=ctx.result.alpha, power=0.8)
    )
    observed = abs(ctx.result.effect.value)
    return [
        Critique(
            check_id="power.underpowered",
            category=CritiqueCategory.power,
            severity=Severity.high,
            title="Underpowered: a null result that proves nothing",
            evidence=(
                f"with n={n1} vs {n2}, the smallest effect detectable at 80% power is "
                f"d={mde:.2f} (a large effect); the observed d={observed:.2f} is far "
                "smaller, so a real moderate effect would likely have been missed"
            ),
            evidence_data={"mde": mde, "observed_d": observed, "n1": n1, "n2": n2},
            remedy=(
                f"do not read this non-significant result as no effect; about "
                f"{int(np.ceil(needed))} per group would be needed to detect a moderate "
                "effect (d=0.5)"
            ),
        )
    ]


@register_check(CritiqueCategory.leakage)
def leakage(ctx: CritiqueContext) -> list[Critique]:
    if ctx.plan.method not in (Method.ols, Method.logistic):
        return []
    leaking = [p for p in ctx.plan.predictors if p in ctx.profile.likely_id_columns]
    if not leaking:
        return []
    target = leaking[0]
    col = ctx.profile.by_name(target)
    card = col.cardinality_fraction if col else 1.0
    return [
        Critique(
            check_id="leakage.id_predictor",
            category=CritiqueCategory.leakage,
            severity=Severity.high,
            title="Identifier used as a predictor",
            evidence=(
                f"{target!r} looks like an identifier (cardinality {card:.0%}); as a "
                "predictor it memorizes rows rather than capturing signal"
            ),
            evidence_data={"predictor": target, "cardinality": card},
            remedy=f"drop {target!r} from the model",
            mechanical_fix=FixAction(
                kind="drop_predictor",
                target=target,
                reason="an identifier carries no generalizable information",
            ),
        )
    ]


@register_check(CritiqueCategory.missing_data)
def missing_data(ctx: CritiqueContext) -> list[Critique]:
    cols = [ctx.plan.outcome, ctx.plan.group, *ctx.plan.predictors]
    out = []
    for name in [c for c in cols if c]:
        col = ctx.profile.by_name(name)
        if col is None or col.missing_fraction <= 0.05:
            continue
        dropped = col.n_missing
        out.append(
            Critique(
                check_id="missing.handling",
                category=CritiqueCategory.missing_data,
                severity=Severity.medium,
                title=f"Missing data dropped from {name!r}",
                evidence=(
                    f"{name!r} is {col.missing_fraction:.0%} missing; listwise deletion "
                    f"dropped {dropped} rows. If those are not missing at random, the "
                    "estimate is biased"
                ),
                evidence_data={
                    "missing_fraction": col.missing_fraction,
                    "dropped": dropped,
                },
                remedy="report the dropped n and consider whether the data is missing at random",
            )
        )
    return out


@register_check(CritiqueCategory.outliers)
def outlier_sensitivity(ctx: CritiqueContext) -> list[Critique]:
    # Only the t-test family, where dropping points and re-running is a clean check and
    # the rank test is the robust fallback if the result hinges on a few observations.
    if ctx.plan.method not in (Method.students_t, Method.welch_t):
        return []
    assert ctx.plan.outcome and ctx.plan.group
    sub = ctx.data[[ctx.plan.outcome, ctx.plan.group]].dropna()
    levels = sorted(sub[ctx.plan.group].dropna().unique().tolist(), key=str)
    if len(levels) != 2:
        return []

    trimmed = [
        _drop_far_out(sub.loc[sub[ctx.plan.group] == lv, ctx.plan.outcome])
        for lv in levels
    ]
    n_dropped = sum(len(sub[sub[ctx.plan.group] == lv]) for lv in levels) - sum(
        t.size for t in trimmed
    )
    if n_dropped == 0 or min(t.size for t in trimmed) < 3:
        return []

    routine = (
        stats_pkg.students_t
        if ctx.plan.method == Method.students_t
        else stats_pkg.welch_t
    )
    refit = routine(trimmed[0], trimmed[1], alpha=ctx.result.alpha)
    flipped = refit.is_significant != ctx.result.is_significant
    if not flipped:
        return []
    return [
        Critique(
            check_id="outliers.sensitivity",
            category=CritiqueCategory.outliers,
            severity=Severity.high,
            title="Result hinges on a few outliers",
            evidence=(
                f"dropping {n_dropped} point(s) beyond 3*IQR flips significance "
                f"(p {ctx.result.p_value:.3f} -> {refit.p_value:.3f}); the conclusion "
                "rests on a handful of observations"
            ),
            evidence_data={
                "n_dropped": n_dropped,
                "p_full": ctx.result.p_value,
                "p_trimmed": refit.p_value,
            },
            remedy="use a rank-based test that is not driven by extreme values",
            mechanical_fix=FixAction(
                kind="switch_method",
                target=Method.mann_whitney_u.value,
                reason="the rank test is robust to the outliers the result depends on",
            ),
        )
    ]


def _drop_far_out(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    iqr = q75 - q25
    # Tukey's "far out" fence (3*IQR), not the milder 1.5: we only care about points
    # extreme enough to plausibly drive the result.
    lower, upper = q25 - 3 * iqr, q75 + 3 * iqr
    kept: np.ndarray = arr[(arr >= lower) & (arr <= upper)]
    return kept
