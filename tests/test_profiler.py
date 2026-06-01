from __future__ import annotations

import numpy as np
import pandas as pd

from statskeptic.profile import ColumnKind, build_profile


def test_column_kinds_are_classified():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "patient_id": range(1, 61),
            "arm": ["ctrl", "treat"] * 30,
            "dose": np.repeat([1, 2, 3, 4, 5, 6], 10),
            "weight": rng.normal(70, 10, 60),
            "note": [f"free text number {i}" for i in range(60)],
            "constant": [7] * 60,
            "visit": pd.to_datetime("2026-01-01")
            + pd.to_timedelta(range(60), unit="D"),
        }
    )
    p = build_profile(df)
    kind = {c.name: c.kind for c in p.columns}
    assert kind["arm"] == ColumnKind.boolean  # two levels
    assert kind["dose"] == ColumnKind.categorical  # few integer codes
    assert kind["weight"] == ColumnKind.numeric
    assert kind["note"] == ColumnKind.text  # high-cardinality free text
    assert kind["constant"] == ColumnKind.constant
    assert kind["visit"] == ColumnKind.datetime
    # patient_id is a near-unique integer named like an identifier.
    assert "patient_id" in p.likely_id_columns


def test_numeric_summary_captures_skew_and_outliers():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 99), [50.0]])  # one planted outlier
    p = build_profile(pd.DataFrame({"x": x}))
    summary = p.by_name("x").numeric
    assert summary is not None
    assert summary.n_outliers_iqr >= 1
    assert summary.skew > 1


def test_correlations_present_between_numeric_columns():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 100)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(0, 0.1, 100)})
    p = build_profile(df)
    assert p.correlation("a", "b") > 0.9
