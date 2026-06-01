"""The typed output of the skeptic: a `Critique` per objection, the `FixAction` that
says whether it can be mechanically resolved, the `RevisionStep` audit trail, and the
final `Verdict`.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..stats.results import JSONScalar, Severity


class CritiqueCategory(str, Enum):
    assumption = "assumption"
    multiple_comparisons = "multiple_comparisons"
    confounding = "confounding"
    power = "power"
    leakage = "leakage"
    outliers = "outliers"
    missing_data = "missing_data"
    scope = "scope"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FixAction(_Frozen):
    """How the revision loop can act on a finding. Present only when the fix is
    mechanical; its absence means the finding is a caveat the loop cannot resolve."""

    kind: Literal["switch_method", "apply_correction", "drop_predictor"]
    target: str
    reason: str


class Critique(_Frozen):
    check_id: str
    category: CritiqueCategory
    severity: Severity
    title: str
    # Drawn from the actual numbers, never generic. This is what makes the objection
    # checkable rather than boilerplate.
    evidence: str
    evidence_data: dict[str, JSONScalar] = {}
    remedy: str
    mechanical_fix: FixAction | None = None


class RevisionStep(_Frozen):
    from_method: str
    to_method: str
    trigger: str
    reason: str
    p_before: float
    p_after: float


class Verdict(str, Enum):
    defensible = "defensible"
    defensible_with_caveats = "defensible_with_caveats"
    cannot_conclude = "cannot_conclude"
    declined = "declined"
