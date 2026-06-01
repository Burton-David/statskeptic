"""The vetted statistical toolset. Every number in a statskeptic report originates
here, in code that is tested against known answers and re-runnable from its own
`Computation` record. The planner selects a routine; it never writes one.
"""

from .association import chi_square, fisher_exact, pearson, spearman
from .k_group import anova_oneway, kruskal
from .regression import logistic, ols
from .results import (
    AssociationResult,
    AssumptionCheck,
    Computation,
    ConfidenceInterval,
    EffectSize,
    KGroupResult,
    RegressionResult,
    Severity,
    StatResult,
    TwoGroupResult,
)
from .two_group import mann_whitney_u, students_t, welch_t

__all__ = [
    "AssociationResult",
    "AssumptionCheck",
    "Computation",
    "ConfidenceInterval",
    "EffectSize",
    "KGroupResult",
    "RegressionResult",
    "Severity",
    "StatResult",
    "TwoGroupResult",
    "anova_oneway",
    "chi_square",
    "fisher_exact",
    "kruskal",
    "logistic",
    "mann_whitney_u",
    "ols",
    "pearson",
    "spearman",
    "students_t",
    "welch_t",
]
