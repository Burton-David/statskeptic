"""statskeptic - a data-analysis agent that distrusts its own conclusions.

Point it at a dataset and a question. It plans an analysis, runs it with real,
vetted statistical code (never a number invented by a language model), then attacks
its own result against a methodological rubric - assumption violations, multiple
comparisons, confounding, underpowered samples, leakage - and reports plainly what
it found and, just as importantly, what it cannot conclude.

The public API grows as each capability lands; for now this exposes the package
version so an editable install resolves cleanly.
"""

from importlib.metadata import PackageNotFoundError, version

from .agent import analyze
from .critique.models import Critique, CritiqueCategory, Verdict
from .errors import AnalysisError, PlanningError, StatskepticError
from .plan.models import AnalysisPlan, Decline, Method, PlanHints, QuestionType
from .profile.models import DataProfile
from .report.models import Analysis, Report

try:
    __version__ = version("statskeptic")
except PackageNotFoundError:  # pragma: no cover - only during local source runs
    __version__ = "0.0.0"

__all__ = [
    "Analysis",
    "AnalysisError",
    "AnalysisPlan",
    "Critique",
    "CritiqueCategory",
    "DataProfile",
    "Decline",
    "Method",
    "PlanHints",
    "PlanningError",
    "QuestionType",
    "Report",
    "StatskepticError",
    "Verdict",
    "__version__",
    "analyze",
]
