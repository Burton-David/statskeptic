"""Shared builders for the planted-trap datasets the critique and agent tests reuse.

Seeded so every expected number is reproducible. These are the same shapes the demo
corpus uses; keeping them here means the tests exercise exactly what the README shows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def skewed_two_group() -> pd.DataFrame:
    # One group is heavily right-skewed (lognormal), so a t-test on the means is the
    # wrong tool and the skeptic should move to Mann-Whitney.
    rng = np.random.default_rng(42)
    ctrl = rng.lognormal(1.0, 0.8, 60)
    treat = rng.lognormal(1.35, 0.8, 60)
    return pd.DataFrame(
        {
            "treatment": ["ctrl"] * 60 + ["treat"] * 60,
            "recovery": np.concatenate([ctrl, treat]),
        }
    )


@pytest.fixture
def screen_many() -> pd.DataFrame:
    # 20 candidate columns, only v0 truly tied to the outcome. Run uncorrected, several
    # will look significant by chance; that is the multiple-comparisons trap.
    rng = np.random.default_rng(7)
    n = 200
    cols = {f"v{i}": rng.normal(0, 1, n) for i in range(20)}
    outcome = 0.45 * cols["v0"] + rng.normal(0, 1, n)
    return pd.DataFrame({**cols, "outcome": outcome})


@pytest.fixture
def confounded() -> pd.DataFrame:
    # exercise and health are linked only through age; a causal reading is unsupported.
    rng = np.random.default_rng(7)
    age = rng.normal(0, 1, 300)
    return pd.DataFrame(
        {
            "exercise": 0.8 * age + rng.normal(0, 0.5, 300),
            "health": 0.8 * age + rng.normal(0, 0.5, 300),
            "age": age,
        }
    )


@pytest.fixture
def underpowered() -> pd.DataFrame:
    # A real moderate effect, but n is far too small to detect it reliably.
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "grp": ["a"] * 9 + ["b"] * 9,
            "y": np.concatenate([rng.normal(0, 1, 9), rng.normal(0.5, 1, 9)]),
        }
    )
