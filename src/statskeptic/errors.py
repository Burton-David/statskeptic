"""Errors statskeptic raises when an analysis cannot be completed honestly.

These are surfaced, never swallowed into a confident-looking answer. The CLI maps them
to a non-zero exit code so a failed analysis is loud, not silent.
"""

from __future__ import annotations


class StatskepticError(Exception):
    """Base class for everything this package raises deliberately."""


class AnalysisError(StatskepticError):
    """A statistical routine could not produce a trustworthy result (singular design,
    perfect separation, a degenerate column). The cause is reported as-is."""


class PlanningError(StatskepticError):
    """The question could not be mapped to a vetted method. Carries the honest decline
    reason and the list of methods that are supported."""

    def __init__(self, reason: str, supported: list[str]):
        super().__init__(reason)
        self.reason = reason
        self.supported = supported
