"""The plan the agent commits to before touching the data: which question this is,
which vetted method answers it, and on which columns.

The planner picks the method a conventional analyst would reach for first, not the
safest one. Upgrading a t-test to Mann-Whitney when the data is skewed is the critique
engine's job; if the planner pre-empted it, the skeptic would have nothing to catch.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class Method(str, Enum):
    students_t = "students_t"
    welch_t = "welch_t"
    mann_whitney_u = "mann_whitney_u"
    pearson = "pearson"
    spearman = "spearman"
    chi_square = "chi_square"
    fisher_exact = "fisher_exact"
    anova_oneway = "anova_oneway"
    kruskal = "kruskal"
    ols = "ols"
    logistic = "logistic"


class QuestionType(str, Enum):
    comparison = "comparison"
    association = "association"
    regression = "regression"
    trend = "trend"
    screen = "screen"


class PlanHints(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    outcome: str | None = None
    group: str | None = None
    predictors: list[str] | None = None


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    question: str
    question_type: QuestionType
    # None only for a screen, where execution chooses a method per candidate column.
    method: Method | None = None
    outcome: str | None = None
    group: str | None = None
    predictors: list[str] = []
    candidates: list[str] = []
    rationale: str = ""
    assumptions: list[str] = []


class Decline(BaseModel):
    """The honest non-answer: the question does not map to a vetted method, or the
    columns could not be resolved. Never an improvised analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    reason: str
    supported: list[str]
