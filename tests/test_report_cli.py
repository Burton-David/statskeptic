from __future__ import annotations

import json

import numpy as np
import pandas as pd

from statskeptic import analyze
from statskeptic.cli import main


def _clean_two_group() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "arm": ["a"] * 50 + ["b"] * 50,
            "score": np.concatenate([rng.normal(0, 1, 50), rng.normal(1.0, 1, 50)]),
        }
    )


def test_explain_always_has_a_cannot_conclude_section(skewed_two_group):
    text = analyze(skewed_two_group, "Does treatment change recovery?").explain()
    assert "## What this cannot conclude" in text


def test_to_json_round_trips(skewed_two_group):
    payload = analyze(skewed_two_group, "Does treatment change recovery?").to_json()
    data = json.loads(payload)
    assert data["verdict"]
    assert data["analyses"][0]["result"]["method"] == "Mann-Whitney U"


def test_cli_exit_zero_on_defensible(tmp_path, capsys):
    path = tmp_path / "clean.csv"
    _clean_two_group().to_csv(path, index=False)
    code = main(["analyze", str(path), "-q", "Does score differ by arm?"])
    assert code == 0


def test_cli_exit_two_on_cannot_conclude(tmp_path):
    path = tmp_path / "small.csv"
    rng = np.random.default_rng(11)
    pd.DataFrame(
        {
            "grp": ["a"] * 9 + ["b"] * 9,
            "y": np.concatenate([rng.normal(0, 1, 9), rng.normal(0.5, 1, 9)]),
        }
    ).to_csv(path, index=False)
    code = main(
        [
            "analyze",
            str(path),
            "-q",
            "Is there a difference in y between the groups?",
            "--quiet",
        ]
    )
    assert code == 2


def test_cli_exit_three_on_declined(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(path, index=False)
    code = main(["analyze", str(path), "-q", "Train a transformer", "--quiet"])
    assert code == 3


def test_cli_exit_usage_on_missing_file(capsys):
    code = main(["analyze", "/no/such/file.csv", "-q", "anything"])
    assert code == 64


def test_cli_json_is_parseable(tmp_path, capsys):
    path = tmp_path / "clean.csv"
    _clean_two_group().to_csv(path, index=False)
    code = main(["analyze", str(path), "-q", "Does score differ by arm?", "--json"])
    assert code == 0
    json.loads(capsys.readouterr().out)
