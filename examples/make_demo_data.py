"""Generate the planted-trap demo corpus.

Each dataset hides one methodological error a fluent tool would walk straight into.
The seeds are fixed, so the committed CSVs and the numbers in the README are
reproducible. Run `python examples/make_demo_data.py` to regenerate them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def skewed_trial() -> pd.DataFrame:
    # Both arms are drawn from the SAME heavy-tailed distribution, so there is no real
    # effect. Seed 35 is one where the long right tail drags the means apart enough that
    # a t-test calls it significant (p=0.014). The trap: that significance is an artifact
    # of skew, and the rank test does not reproduce it.
    rng = np.random.default_rng(35)
    placebo = rng.lognormal(3.0, 1.1, 45)
    drug = rng.lognormal(3.0, 1.1, 45)
    return pd.DataFrame(
        {
            "arm": ["placebo"] * 45 + ["drug"] * 45,
            "recovery_hours": np.round(np.concatenate([placebo, drug]), 1),
        }
    )


def biomarker_screen() -> pd.DataFrame:
    # 24 candidate markers tested against one outcome; only marker_00 carries a real
    # signal. At seed 0 three of the noise markers (11, 19, 21) also clear p<0.05 by
    # chance. The trap: report them uncorrected and you ship three false discoveries.
    rng = np.random.default_rng(0)
    n = 140
    markers = {f"marker_{i:02d}": rng.normal(0, 1, n) for i in range(24)}
    outcome = 0.40 * markers["marker_00"] + rng.normal(0, 1, n)
    frame = {k: np.round(v, 3) for k, v in markers.items()}
    frame["outcome"] = np.round(outcome, 3)
    return pd.DataFrame(frame)


def exercise_health() -> pd.DataFrame:
    # exercise and health are correlated only because both rise with age; there is no
    # direct link. A causal question on this observational data is unsupported.
    rng = np.random.default_rng(7)
    age = rng.normal(45, 12, 300)
    exercise = 0.5 * (age - 45) + rng.normal(0, 4, 300)
    health = -0.4 * (age - 45) + rng.normal(0, 4, 300)
    return pd.DataFrame(
        {
            "weekly_exercise_hours": np.round(exercise - exercise.min(), 2),
            "health_score": np.round(health - health.min(), 2),
            "age": np.round(age, 1),
        }
    )


def small_trial() -> pd.DataFrame:
    # A real moderate effect (d=0.5), but nine per arm cannot reliably detect it. The
    # non-significant result says nothing; it is not evidence of no effect.
    rng = np.random.default_rng(11)
    control = rng.normal(100, 15, 9)
    treated = rng.normal(107.5, 15, 9)
    return pd.DataFrame(
        {
            "group": ["control"] * 9 + ["treated"] * 9,
            "test_score": np.round(np.concatenate([control, treated]), 1),
        }
    )


def clean_ab_test() -> pd.DataFrame:
    # Clean, roughly normal, a genuine difference, ample n: a defensible result with
    # nothing for the skeptic to overturn. The honest "yes" case.
    rng = np.random.default_rng(4)
    a = rng.normal(50, 12, 200)
    b = rng.normal(56, 12, 200)
    return pd.DataFrame(
        {
            "variant": ["A"] * 200 + ["B"] * 200,
            "order_value": np.round(np.concatenate([a, b]), 2),
        }
    )


_DATASETS = {
    "skewed_trial.csv": skewed_trial,
    "biomarker_screen.csv": biomarker_screen,
    "exercise_health.csv": exercise_health,
    "small_trial.csv": small_trial,
    "clean_ab_test.csv": clean_ab_test,
}


def main() -> None:
    for name, builder in _DATASETS.items():
        path = HERE / name
        builder().to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
