"""The honest report. It states what was done, the result with effect size and CI, the
assumptions and whether they held, the objections the skeptic raised and how each was
resolved, and an explicit "what this cannot conclude" section that is never dropped to
make the result look tidier.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..critique.models import Critique, RevisionStep, Verdict
from ..plan.models import AnalysisPlan, Decline
from ..stats.results import RegressionResult, StatResult


class Analysis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    plan: AnalysisPlan
    result: StatResult
    revision_steps: list[RevisionStep] = []
    critiques: list[Critique] = []
    verdict: Verdict


class Report(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    question: str
    n_rows: int
    n_cols: int
    analyses: list[Analysis] = []
    session_critiques: list[Critique] = []
    verdict: Verdict
    cannot_conclude: list[str] = []
    declined: Decline | None = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def explain(self) -> str:
        return render_markdown(self)


_VERDICT_LABEL = {
    Verdict.defensible: "Defensible result",
    Verdict.defensible_with_caveats: "Defensible, with caveats",
    Verdict.cannot_conclude: "Cannot conclude a reliable answer",
    Verdict.declined: "Declined: no vetted method fits this question",
}


def _fmt(x: float | None, places: int = 4) -> str:
    if x is None:
        return "n/a"
    if x != x:  # NaN
        return "nan"
    if x in (float("inf"), float("-inf")):
        return "inf"
    return f"{x:.{places}g}"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def render_markdown(report: Report) -> str:
    lines: list[str] = ["# statskeptic report", ""]
    lines.append(f"**Question:** {report.question}")
    lines.append("")
    lines.append(f"**Verdict:** {_VERDICT_LABEL[report.verdict]}")
    lines.append("")
    lines.append(f"_Data: {report.n_rows} rows, {report.n_cols} columns._")
    lines.append("")

    if report.declined is not None:
        lines.append("## Declined")
        lines.append(report.declined.reason)
        lines.append("")
        lines.append("statskeptic can answer:")
        lines.extend(f"- {s}" for s in report.declined.supported)
        lines.append("")
        return "\n".join(lines)

    if report.session_critiques:
        lines.extend(_render_screen(report))
    else:
        for analysis in report.analyses:
            lines.extend(_render_analysis(analysis))

    lines.append("## What this cannot conclude")
    if report.cannot_conclude:
        lines.extend(f"- {c}" for c in report.cannot_conclude)
    else:
        lines.append("- Nothing beyond the assumptions and caveats noted above.")
    lines.append("")
    return "\n".join(lines)


def _render_analysis(analysis: Analysis) -> list[str]:
    r = analysis.result
    out = [f"## {r.method}", analysis.plan.rationale, ""]

    sig = "significant" if r.is_significant else "not significant"
    p_text = _fmt_p(r.p_value)
    if r.p_value_corrected is not None:
        p_text += f" ({r.correction} corrected p={_fmt_p(r.p_value_corrected)})"
    out.append(
        f"- Result: {r.statistic_name} = {_fmt(r.statistic)}, p = {p_text} ({sig} at alpha={r.alpha})"
    )

    eff = r.effect
    eff_line = f"- Effect: {eff.name} = {_fmt(eff.value)}"
    if eff.ci_low is not None and eff.ci_level is not None:
        eff_line += (
            f", {int(eff.ci_level * 100)}% CI [{_fmt(eff.ci_low)}, {_fmt(eff.ci_high)}]"
        )
    if eff.interpretation:
        eff_line += f" ({eff.interpretation})"
    out.append(eff_line)

    if r.ci is not None:
        out.append(
            f"- {r.ci.quantity}: {int(r.ci.level * 100)}% CI "
            f"[{_fmt(r.ci.low)}, {_fmt(r.ci.high)}]"
        )
    out.append(f"- n = {r.n}")
    out.append("")

    if isinstance(r, RegressionResult):
        out.extend(_render_regression(r))

    out.append("### Assumptions checked")
    out.append("| check | holds | detail |")
    out.append("| --- | --- | --- |")
    for c in r.assumptions:
        out.append(f"| {c.name} | {'yes' if c.holds else 'NO'} | {c.detail} |")
    out.append("")

    out.append("### Revisions")
    if analysis.revision_steps:
        for s in analysis.revision_steps:
            out.append(
                f"- Switched from {s.from_method} to {s.to_method} ({s.trigger}): "
                f"{s.reason}. p {_fmt_p(s.p_before)} -> {_fmt_p(s.p_after)}."
            )
    else:
        out.append("- None needed.")
    out.append("")

    out.append("### Objections raised")
    if analysis.critiques:
        for crit in analysis.critiques:
            out.append(
                f"- [{crit.severity.value}] {crit.title}: {crit.evidence} "
                f"Remedy: {crit.remedy}"
            )
    else:
        out.append("- None outstanding.")
    out.append("")
    return out


def _render_regression(r: RegressionResult) -> list[str]:
    unit = "odds ratio" if r.model_type == "logistic" else "coefficient"
    out = [
        f"### Coefficients ({unit})",
        f"| term | {unit} | p | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for name, value in r.params.items():
        ci = r.param_cis.get(name)
        ci_text = f"[{_fmt(ci.low)}, {_fmt(ci.high)}]" if ci else "n/a"
        out.append(
            f"| {name} | {_fmt(value)} | {_fmt_p(r.param_pvalues.get(name))} | {ci_text} |"
        )
    out.append("")
    return out


def _render_screen(report: Report) -> list[str]:
    out = ["## Screen across candidate variables", ""]
    mc = next(
        (c for c in report.session_critiques if c.check_id == "mc.uncorrected"), None
    )
    corrected = any(a.result.correction for a in report.analyses)
    if mc is not None:
        out.append(f"**{mc.title}.** {mc.evidence}")
        out.append("")
    if corrected:
        out.append(
            "Holm correction applied across the family; significance re-read below."
        )
        out.append("")

    out.append("| variable | statistic | raw p | corrected p | significant |")
    out.append("| --- | --- | --- | --- | --- |")
    for a in report.analyses:
        r = a.result
        var = a.plan.predictors[0] if a.plan.predictors else a.plan.outcome
        corr = _fmt_p(r.p_value_corrected) if r.p_value_corrected is not None else "n/a"
        out.append(
            f"| {var} | {_fmt(r.statistic)} | {_fmt_p(r.p_value)} | {corr} | "
            f"{'yes' if r.is_significant else 'no'} |"
        )
    out.append("")
    return out
