"""Errors statskeptic raises when an analysis cannot be completed honestly.

These are surfaced, never swallowed into a confident-looking answer. The CLI maps them
to a non-zero exit code so a failed analysis is loud, not silent.
"""

from __future__ import annotations


class StatskepticError(Exception):
    """Base class for everything this package raises deliberately."""


class AnalysisError(StatskepticError):
    """A statistical routine could not produce a trustworthy result (singular design,
    perfect separation, a degenerate column). The cause is reported as-is.

    A question that does not map to a vetted method is not an error: the planner returns
    a `Decline` value and the report says so. This is reserved for data that defeats a
    chosen method."""
