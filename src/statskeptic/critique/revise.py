"""The revision loop: take a critiqued analysis and actually fix what can be fixed.

High-severity findings with a mechanical remedy drive a re-run (switch to a rank test,
switch to Welch, drop a leaking predictor); findings that cannot be fixed by re-running
(confounding, low power) survive as caveats and push the verdict to "cannot conclude".
The loop records every change it makes, because "it caught and corrected its own
mistake" is only a credible claim if the trail is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from statsmodels.stats.multitest import multipletests

from ..execution import execute
from ..plan.models import AnalysisPlan, Method
from ..stats.results import Severity, StatResult
from .engine import CritiqueContext, run_critique
from .models import Critique, CritiqueCategory, FixAction, RevisionStep, Verdict

if TYPE_CHECKING:
    import pandas as pd

    from ..profile.models import DataProfile


@dataclass
class RevisionOutcome:
    plan: AnalysisPlan
    result: StatResult
    steps: list[RevisionStep] = field(default_factory=list)
    residual: list[Critique] = field(default_factory=list)


# Lower runs first. Drop a leak before choosing a test; fix normality before variance
# (the rank test makes the variance question moot); reach for the robust test last.
_FIX_ORDER = {
    "leakage.id_predictor": 0,
    "assumption.expected_counts": 1,
    "assumption.normality": 2,
    "assumption.equal_variance": 3,
    "outliers.sensitivity": 4,
}

_HIGH = (Severity.high, Severity.critical)


def revise(
    plan: AnalysisPlan,
    result: StatResult,
    profile: DataProfile,
    data: pd.DataFrame,
    alpha: float = 0.05,
    max_iters: int = 4,
) -> RevisionOutcome:
    steps: list[RevisionStep] = []
    applied: set[tuple[str, str]] = set()
    current_plan, current = plan, result

    for _ in range(max_iters):
        critiques = run_critique(CritiqueContext(current_plan, current, profile, data))
        actionable = [
            c
            for c in critiques
            if c.severity in _HIGH
            and c.mechanical_fix is not None
            and (c.check_id, c.mechanical_fix.target) not in applied
        ]
        if not actionable:
            break
        chosen = min(
            actionable, key=lambda c: (_FIX_ORDER.get(c.check_id, 99), -c.severity.rank)
        )
        assert chosen.mechanical_fix is not None
        applied.add((chosen.check_id, chosen.mechanical_fix.target))
        outcome = _apply_fix(chosen.mechanical_fix, current_plan, data, alpha)
        if outcome is None:
            continue
        new_plan, new_result = outcome
        steps.append(
            RevisionStep(
                from_method=current.method,
                to_method=new_result.method,
                trigger=chosen.check_id,
                reason=chosen.mechanical_fix.reason,
                p_before=current.p_value,
                p_after=new_result.p_value,
            )
        )
        current_plan, current = new_plan, new_result

    residual = run_critique(CritiqueContext(current_plan, current, profile, data))
    return RevisionOutcome(
        plan=current_plan, result=current, steps=steps, residual=residual
    )


def _apply_fix(
    fix: FixAction, plan: AnalysisPlan, data: pd.DataFrame, alpha: float
) -> tuple[AnalysisPlan, StatResult] | None:
    if fix.kind == "switch_method":
        new_plan = plan.model_copy(update={"method": Method(fix.target)})
        return new_plan, execute(new_plan, data, alpha)
    if fix.kind == "drop_predictor":
        remaining = [p for p in plan.predictors if p != fix.target]
        if not remaining:
            # Nothing left to model; leave the finding as an unresolved caveat.
            return None
        new_plan = plan.model_copy(update={"predictors": remaining})
        return new_plan, execute(new_plan, data, alpha)
    return None


def multiple_comparisons_critique(
    results: list[StatResult], outcome: str, alpha: float
) -> Critique | None:
    if len(results) < 2 or any(r.correction for r in results):
        return None
    pvals = [r.p_value for r in results]
    n = len(pvals)
    n_sig = sum(p < alpha for p in pvals)
    expected_false = n * alpha
    return Critique(
        check_id="mc.uncorrected",
        category=CritiqueCategory.multiple_comparisons,
        severity=Severity.high,
        title="Multiple comparisons without correction",
        evidence=(
            f"{n} tests against {outcome!r}; {n_sig} reach p<{alpha} uncorrected, but "
            f"about {expected_false:.1f} would by chance alone. Some 'findings' are "
            "likely false positives"
        ),
        evidence_data={
            "n_tests": n,
            "n_significant_raw": n_sig,
            "expected_false": expected_false,
        },
        remedy="apply a multiplicity correction (Holm) and re-read significance",
        mechanical_fix=FixAction(
            kind="apply_correction",
            target="holm",
            reason="controlling the family-wise error rate across the screen",
        ),
    )


def apply_holm(results: list[StatResult], alpha: float) -> list[StatResult]:
    """Holm-Bonferroni across a family. It changes only the significance decision, never
    a test statistic, so the underlying numbers stay intact and re-runnable."""
    pvals = [r.p_value for r in results]
    _, p_corrected, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return [
        r.model_copy(update={"p_value_corrected": float(pc), "correction": "holm"})
        for r, pc in zip(results, p_corrected, strict=True)
    ]


def decide_verdict(residual: list[Critique]) -> Verdict:
    severities = [c.severity for c in residual]
    if any(s in _HIGH for s in severities):
        # A surviving high-severity objection means we could not make the result
        # defensible. Saying so is the honest outcome, not a failure of the tool.
        return Verdict.cannot_conclude
    if any(s == Severity.medium for s in severities):
        return Verdict.defensible_with_caveats
    return Verdict.defensible
