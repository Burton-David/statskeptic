"""Rule-based planner: a natural-language question plus a `DataProfile` into an
`AnalysisPlan`, with no model in the loop.

A keyword-and-structure planner is unavoidably heuristic. The design choice that keeps
it honest is that it declines when it cannot resolve the columns confidently, naming the
candidates and the flag to disambiguate, rather than guessing and producing a confident
analysis of the wrong variables.
"""

from __future__ import annotations

import re

from ..profile.models import ColumnKind, ColumnProfile, DataProfile
from .models import AnalysisPlan, Decline, Method, PlanHints, QuestionType

SUPPORTED = [
    "two-group comparison (t-test / Mann-Whitney)",
    "k-group comparison (ANOVA / Kruskal-Wallis)",
    "association (Pearson / Spearman / chi-square)",
    "regression (OLS / logistic)",
    "screen many variables against one outcome",
]

_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "is",
    "are",
    "do",
    "does",
    "did",
    "and",
    "or",
    "with",
    "on",
    "in",
    "by",
    "for",
    "vs",
    "versus",
    "between",
    "there",
    "any",
    "have",
    "has",
    "this",
    "that",
    "be",
    "it",
    "as",
    "at",
    "from",
    "than",
}

_REGRESSION_CUES = ("predict", "regress", "control for", "adjust", "driver", "explain")
_TREND_CUES = ("over time", "trend", "since", "increase", "decline", "decrease")
_COMPARISON_CUES = (
    "differ",
    "difference",
    "compare",
    "between",
    "higher",
    "lower",
    "affect",
    "effect of",
    "impact of",
    "cause",
    "more",
    "less",
    "change",
    "reduce",
    "increase",
    "decrease",
    "improve",
    "raise",
    "boost",
    "worsen",
)
_ASSOCIATION_CUES = ("relationship", "associat", "correlat", "related to", "linked")
# A screen is a "which/what of these many things relates to X" question. The noun varies
# by domain (markers, factors, features), so rather than enumerate nouns we key on the
# wh-word plus an association/prediction cue and let the candidate count confirm it.
_SCREEN_WH = ("which", "what")
_SCREEN_CUES = (
    "associat",
    "correlat",
    "related",
    "predict",
    "linked",
    "driver",
    "factor",
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(text: str) -> set[str]:
    spaced = _CAMEL.sub(" ", text)
    raw = re.split(r"[^a-zA-Z0-9]+", spaced.lower())
    return {t for t in raw if len(t) >= 2 and t not in _STOPWORDS}


def _match_score(col: ColumnProfile, q_tokens: set[str]) -> int:
    col_tokens = _tokens(col.name)
    return len(col_tokens & q_tokens)


def _usable(col: ColumnProfile) -> bool:
    return not col.is_likely_id and col.kind != ColumnKind.constant


def plan(
    question: str, profile: DataProfile, hints: PlanHints | None = None
) -> AnalysisPlan | Decline:
    hints = hints or PlanHints()
    q = question.lower()
    q_tokens = _tokens(question)

    scored = sorted(
        ((c, _match_score(c, q_tokens)) for c in profile.columns),
        key=lambda cs: cs[1],
        reverse=True,
    )
    named = [c for c, s in scored if s > 0]

    qtype = _classify(q, named, profile, hints)
    if qtype is None:
        return Decline(
            reason=(
                "could not tell what kind of question this is (comparison, "
                "association, regression, or trend) from the wording or the columns"
            ),
            supported=SUPPORTED,
        )

    builders = {
        QuestionType.screen: _plan_screen,
        QuestionType.regression: _plan_regression,
        QuestionType.trend: _plan_trend,
        QuestionType.comparison: _plan_comparison,
        QuestionType.association: _plan_association,
    }
    return builders[qtype](question, profile, hints, named)


def _classify(
    q: str, named: list[ColumnProfile], profile: DataProfile, hints: PlanHints
) -> QuestionType | None:
    q_tokens = _tokens(q)
    wh = bool(q_tokens & set(_SCREEN_WH))
    # Screen only when there is genuinely a field of candidates to sweep; otherwise a
    # "which" question between two columns is just an ordinary association.
    n_numeric_total = len([c for c in profile.numeric() if _usable(c)])
    if wh and any(c in q for c in _SCREEN_CUES) and n_numeric_total >= 3:
        return QuestionType.screen
    if hints.predictors or any(c in q for c in _REGRESSION_CUES):
        return QuestionType.regression
    has_datetime = any(c.kind == ColumnKind.datetime for c in profile.columns)
    if has_datetime and any(c in q for c in _TREND_CUES):
        return QuestionType.trend
    if any(c in q for c in _COMPARISON_CUES):
        return QuestionType.comparison
    if any(c in q for c in _ASSOCIATION_CUES):
        return QuestionType.association
    # No verbal cue. Fall back to the shape of the columns the question names.
    kinds = [c.kind for c in named[:2]]
    n_numeric = sum(k == ColumnKind.numeric for k in kinds)
    if n_numeric == 2:
        return QuestionType.association
    if n_numeric == 1 and len(kinds) == 2:
        return QuestionType.comparison
    if len(kinds) == 2:
        return QuestionType.association
    return None


def _resolve(
    hint: str | None,
    named: list[ColumnProfile],
    profile: DataProfile,
    kinds: tuple[ColumnKind, ...],
    exclude: set[str],
) -> ColumnProfile | None:
    if hint:
        col = profile.by_name(hint)
        return col
    by_name = next(
        (c for c in named if c.kind in kinds and c.name not in exclude and _usable(c)),
        None,
    )
    if by_name:
        return by_name
    # Profile fallback: if exactly one usable column of the wanted kind exists, it is
    # unambiguous even when the question did not name it.
    candidates = [
        c
        for c in profile.columns
        if c.kind in kinds and c.name not in exclude and _usable(c)
    ]
    return candidates[0] if len(candidates) == 1 else None


_NUMERIC = (ColumnKind.numeric,)
_CATEGORICAL = (ColumnKind.categorical, ColumnKind.boolean)


def _plan_comparison(
    question: str, profile: DataProfile, hints: PlanHints, named: list[ColumnProfile]
) -> AnalysisPlan | Decline:
    outcome = _resolve(hints.outcome, named, profile, _NUMERIC, set())
    group = _resolve(
        hints.group, named, profile, _CATEGORICAL, {outcome.name} if outcome else set()
    )

    if outcome is None or group is None:
        # We could not form a clean numeric-outcome + categorical-group pair. Fall back to
        # the profile's columns (the question may not have named them, e.g. "smokers" vs a
        # column "smoker"): two categoricals -> chi-square, two numerics -> association.
        # A causal question between two numerics ("does X cause Y") lands here too, so the
        # confounding check still gets its say instead of an unhelpful decline.
        named_cats = [c for c in named if c.kind in _CATEGORICAL and _usable(c)]
        cats = (
            named_cats
            if len(named_cats) >= 2
            else [c for c in profile.categorical() if _usable(c)]
        )
        named_nums = [c for c in named if c.kind == ColumnKind.numeric and _usable(c)]
        nums = (
            named_nums
            if len(named_nums) >= 2
            else [c for c in profile.numeric() if _usable(c)]
        )

        if outcome is None and len(cats) >= 2:
            return _association_plan(
                question, QuestionType.comparison, cats[0], cats[1]
            )
        if len(nums) >= 2:
            return _association_plan(
                question, QuestionType.association, nums[0], nums[1]
            )
        if len(cats) >= 2:
            return _association_plan(
                question, QuestionType.comparison, cats[0], cats[1]
            )
        return Decline(
            reason=_unresolved_reason(outcome, group, "a numeric outcome and a group"),
            supported=SUPPORTED,
        )

    n_levels = group.n_unique
    if n_levels == 2:
        method, why = (
            Method.students_t,
            "two groups, so a t-test is the usual first pass",
        )
    else:
        method, why = (
            Method.anova_oneway,
            f"{n_levels} groups, so one-way ANOVA is the usual first pass",
        )
    return AnalysisPlan(
        question=question,
        question_type=QuestionType.comparison,
        method=method,
        outcome=outcome.name,
        group=group.name,
        rationale=f"comparing {outcome.name!r} across {group.name!r}: {why}",
        assumptions=_assumptions_for(method),
    )


def _plan_association(
    question: str, profile: DataProfile, hints: PlanHints, named: list[ColumnProfile]
) -> AnalysisPlan | Decline:
    usable = [c for c in named if _usable(c)]
    if hints.outcome and hints.predictors:
        a = profile.by_name(hints.outcome)
        b = profile.by_name(hints.predictors[0])
        if a and b:
            return _association_plan(question, QuestionType.association, a, b)
    if len(usable) < 2:
        # Two numerics with nothing named: only unambiguous if there are exactly two.
        nums = [c for c in profile.numeric() if _usable(c)]
        if len(nums) == 2:
            return _association_plan(
                question, QuestionType.association, nums[0], nums[1]
            )
        return Decline(
            reason="need two columns to test an association; could not resolve them",
            supported=SUPPORTED,
        )
    return _association_plan(question, QuestionType.association, usable[0], usable[1])


def _association_plan(
    question: str,
    qtype: QuestionType,
    a: ColumnProfile,
    b: ColumnProfile,
) -> AnalysisPlan:
    a_num = a.kind == ColumnKind.numeric
    b_num = b.kind == ColumnKind.numeric
    if a_num and b_num:
        method = Method.pearson
        rationale = (
            f"two numeric variables, {a.name!r} and {b.name!r}: Pearson by default"
        )
    elif not a_num and not b_num:
        method = Method.chi_square
        rationale = f"two categorical variables, {a.name!r} and {b.name!r}: chi-square"
    else:
        # One numeric, one categorical is a group comparison wearing association clothes.
        num, cat = (a, b) if a_num else (b, a)
        if cat.n_unique == 2:
            return AnalysisPlan(
                question=question,
                question_type=QuestionType.comparison,
                method=Method.students_t,
                outcome=num.name,
                group=cat.name,
                rationale=(
                    f"{num.name!r} against the two levels of {cat.name!r} is a "
                    "two-group comparison"
                ),
                assumptions=_assumptions_for(Method.students_t),
            )
        return AnalysisPlan(
            question=question,
            question_type=QuestionType.comparison,
            method=Method.anova_oneway,
            outcome=num.name,
            group=cat.name,
            rationale=f"{num.name!r} across the {cat.n_unique} levels of {cat.name!r}",
            assumptions=_assumptions_for(Method.anova_oneway),
        )
    return AnalysisPlan(
        question=question,
        question_type=qtype,
        method=method,
        outcome=a.name,
        predictors=[b.name],
        rationale=rationale,
        assumptions=_assumptions_for(method),
    )


def _plan_regression(
    question: str, profile: DataProfile, hints: PlanHints, named: list[ColumnProfile]
) -> AnalysisPlan | Decline:
    outcome = _resolve(hints.outcome, named, profile, (*_NUMERIC, *_CATEGORICAL), set())
    if outcome is None:
        return Decline(
            reason="could not resolve the outcome to regress; name it with --outcome",
            supported=SUPPORTED,
        )
    binary_outcome = outcome.kind == ColumnKind.boolean or outcome.n_unique == 2
    if outcome.kind != ColumnKind.numeric and not binary_outcome:
        # A nominal outcome with three or more levels needs multinomial regression, which
        # is not in the v1 toolset. Running OLS on the category codes would be the kind of
        # confident nonsense this tool exists to refuse.
        return Decline(
            reason=(
                f"{outcome.name!r} has {outcome.n_unique} categories; regression here "
                "supports a numeric outcome (OLS) or a two-level outcome (logistic), not "
                "a multi-category one"
            ),
            supported=SUPPORTED,
        )
    if hints.predictors:
        predictors = [p for p in hints.predictors if profile.by_name(p)]
    else:
        predictors = [
            c.name for c in profile.numeric() if c.name != outcome.name and _usable(c)
        ]
    if not predictors:
        return Decline(
            reason="no usable numeric predictors found; name them with --predictors",
            supported=SUPPORTED,
        )

    method = Method.logistic if binary_outcome else Method.ols
    kind_word = "binary" if binary_outcome else "continuous"
    return AnalysisPlan(
        question=question,
        question_type=QuestionType.regression,
        method=method,
        outcome=outcome.name,
        predictors=predictors,
        rationale=(
            f"{kind_word} outcome {outcome.name!r} on {len(predictors)} predictor(s): "
            f"{'logistic' if binary_outcome else 'OLS'} regression"
        ),
        assumptions=_assumptions_for(method),
    )


def _plan_trend(
    question: str, profile: DataProfile, hints: PlanHints, named: list[ColumnProfile]
) -> AnalysisPlan | Decline:
    time_col = next((c for c in profile.columns if c.kind == ColumnKind.datetime), None)
    outcome = _resolve(hints.outcome, named, profile, _NUMERIC, set())
    if time_col is None or outcome is None:
        return Decline(
            reason="a trend needs a datetime column and a numeric outcome",
            supported=SUPPORTED,
        )
    return AnalysisPlan(
        question=question,
        question_type=QuestionType.trend,
        method=Method.ols,
        outcome=outcome.name,
        predictors=[time_col.name],
        rationale=f"linear trend of {outcome.name!r} over {time_col.name!r}",
        assumptions=_assumptions_for(Method.ols),
    )


def _plan_screen(
    question: str, profile: DataProfile, hints: PlanHints, named: list[ColumnProfile]
) -> AnalysisPlan | Decline:
    outcome = _resolve(hints.outcome, named, profile, _NUMERIC, set())
    if outcome is None:
        return Decline(
            reason="a screen needs one numeric outcome to test the candidates against",
            supported=SUPPORTED,
        )
    candidates = [
        c.name for c in profile.numeric() if c.name != outcome.name and _usable(c)
    ]
    if len(candidates) < 2:
        return Decline(
            reason="a screen needs at least two candidate variables",
            supported=SUPPORTED,
        )
    return AnalysisPlan(
        question=question,
        question_type=QuestionType.screen,
        method=None,
        outcome=outcome.name,
        candidates=candidates,
        rationale=(
            f"screening {len(candidates)} variables against {outcome.name!r}; testing "
            "this many at once demands a multiplicity correction"
        ),
        assumptions=["independent observations", "per-test association assumptions"],
    )


def _unresolved_reason(
    outcome: ColumnProfile | None, group: ColumnProfile | None, need: str
) -> str:
    missing = []
    if outcome is None:
        missing.append("outcome")
    if group is None:
        missing.append("group")
    return (
        f"could not resolve the {' and '.join(missing)} for a comparison (need {need}); "
        "name them with --outcome and --group"
    )


_ASSUMPTIONS: dict[Method, list[str]] = {
    Method.students_t: [
        "normality within each group",
        "equal variance between groups",
        "independent observations",
    ],
    Method.welch_t: ["normality within each group", "independent observations"],
    Method.mann_whitney_u: [
        "independent observations",
        "similar distribution shapes for a median reading",
    ],
    Method.pearson: [
        "bivariate normality",
        "linear relationship",
        "independent observations",
    ],
    Method.spearman: ["monotonic relationship", "independent observations"],
    Method.chi_square: ["expected cell counts at least 5", "independent observations"],
    Method.anova_oneway: [
        "normality within each group",
        "homogeneity of variance",
        "independent observations",
    ],
    Method.kruskal: [
        "independent observations",
        "similar shapes for a median reading",
    ],
    Method.ols: [
        "linearity",
        "residual normality",
        "homoscedasticity",
        "no severe multicollinearity",
        "independent observations",
    ],
    Method.logistic: [
        "no separation",
        "adequate events per predictor",
        "independent observations",
    ],
}


def _assumptions_for(method: Method) -> list[str]:
    return list(_ASSUMPTIONS[method])
