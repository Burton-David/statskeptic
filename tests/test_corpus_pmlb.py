"""Robustness sweep over real PMLB datasets. Opt-in and network-dependent, so it is
marked slow and excluded from the default gate; run with `pytest -m slow`.

The contract under test is not that statskeptic picks the textbook-perfect method for
every real dataset (it cannot, with a rule-based planner), but that it never crashes,
never reports a non-finite number as a result, always renders, and always lands on a
legal verdict. Exploratory sweeps over ~125 datasets found zero violations; this pins a
deterministic subset as a standing guard.
"""

from __future__ import annotations

import math

import pytest

from statskeptic import PlanHints, Verdict, analyze
from statskeptic.errors import StatskepticError

pytestmark = pytest.mark.slow

pmlb = pytest.importorskip("pmlb")

# A deterministic slice of the sorted catalog, capped so the download stays tractable.
_CANDIDATES = sorted(pmlb.dataset_names)[::18][:12]

_QUESTIONS = [
    ("Predict the target from the other variables", {"outcome": "target"}),
    ("Which factors are associated with the target?", {}),
]


def _assert_report_is_sound(report) -> None:
    assert report.verdict in set(Verdict)
    # Both renderings must succeed for any legal report.
    assert report.to_json()
    assert report.explain()
    for analysis in report.analyses:
        r = analysis.result
        # No fabricated or degenerate figure may pass as a result.
        assert math.isfinite(r.p_value), f"non-finite p in {r.method}"
        assert math.isfinite(r.effect.value), f"non-finite effect in {r.method}"
        # Every result carries the computation that produced it (traceability).
        assert r.computation.callable and r.computation.library_version


@pytest.mark.parametrize("name", _CANDIDATES)
def test_real_dataset_is_handled_without_crashing(name):
    try:
        df = pmlb.fetch_data(name)
    except Exception as exc:  # noqa: BLE001 - network/availability is not under test
        pytest.skip(f"could not fetch {name}: {exc}")
    if len(df) > 3000:
        df = df.sample(3000, random_state=0)

    for question, hints in _QUESTIONS:
        try:
            report = analyze(df, question, hints=PlanHints(**hints) if hints else None)
        except StatskepticError:
            # An honest refusal on data that defeats the chosen method is acceptable;
            # a multiclass target with no multinomial method is the common case here.
            continue
        _assert_report_is_sound(report)
