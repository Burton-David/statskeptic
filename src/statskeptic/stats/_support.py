"""Plumbing shared by every vetted routine: warning capture, input hashing, and the
reproducibility record.

The warning capture is not optional housekeeping. The test suite runs with
`filterwarnings = error`, so a Shapiro-on-large-n warning, a Mann-Whitney all-ties
warning, or a logistic convergence warning would otherwise abort a perfectly valid
analysis. Capturing them is also the methodologically right thing: a convergence
warning is a finding the skeptic should surface, not noise to swallow.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from .results import Computation, JSONScalar


@contextmanager
def captured_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Record warnings instead of raising them, so a routine can convert a warning
    into a structured assumption check rather than die under filterwarnings=error."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record


def messages(record: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in record]


def digest_inputs(*arrays: np.ndarray) -> str:
    """A content hash of the inputs so a re-run can prove it used the same data.

    Canonicalize to float64 bytes plus shape; reproducibility is only a real claim if
    'same data' is verifiable, not asserted.
    """
    h = hashlib.sha256()
    for a in arrays:
        arr = np.asarray(a, dtype=float)
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _library_version(library: str) -> str:
    if library == "scipy":
        import scipy

        return str(scipy.__version__)
    if library == "statsmodels":
        import statsmodels

        return str(statsmodels.__version__)
    if library == "numpy":
        return str(np.__version__)
    raise ValueError(f"no version lookup for library {library!r}")


def computation(
    callable_name: str,
    args: dict[str, JSONScalar],
    library: str,
    *inputs: np.ndarray,
    code_snippet: str | None = None,
) -> Computation:
    return Computation(
        callable=callable_name,
        args=args,
        library=library,
        library_version=_library_version(library),
        input_digest=digest_inputs(*inputs) if inputs else None,
        code_snippet=code_snippet,
    )
