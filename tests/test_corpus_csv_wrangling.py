"""Loader robustness over the CleverCSV "messy CSV" corpus (van den Burg et al.): real
files from GitHub with non-standard delimiters, quoting, and encodings. Opt-in and
network-dependent, so marked slow; run with `pytest -m slow`.

The contract under test is exactly read_table's: it either returns a parsed table or
refuses with a ValueError, and no other exception may escape it. (Analysis robustness on
real data is covered deterministically by the offline loader test and the PMLB sweep, so
it is kept out of this network test to avoid flaking on partial downloads.) Exploratory
sweeps over ~160 corpus files found zero loader crashes; this samples a handful as a
standing guard.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from statskeptic.loader import read_table

# ResourceWarning here is urllib's SSL socket being finalized by the GC, not anything in
# statskeptic; under the project-wide filterwarnings=error it would otherwise fail a
# random later test. The default (offline) gate keeps the strict filter for library code.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

_URL_LIST = (
    "https://raw.githubusercontent.com/alan-turing-institute/"
    "CSV_Wrangling/master/urls_github.json"
)


def _corpus_sample(n: int = 12) -> list[str]:
    try:
        # Context-managed so the SSL socket is closed; a leaked socket raises a
        # ResourceWarning that filterwarnings=error would pin on an unrelated test.
        with urllib.request.urlopen(_URL_LIST, timeout=30) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - availability is not what we test
        pytest.skip(f"corpus URL list unavailable: {exc}")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    # Deterministic spread across the catalog.
    return [records[i]["urls"][0] for i in range(0, len(records), len(records) // n)][
        :n
    ]


@pytest.mark.parametrize("url", _corpus_sample())
def test_messy_corpus_file_loads_or_refuses_cleanly(url, tmp_path):
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read(3_000_000)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"file unavailable: {exc}")

    path = tmp_path / "corpus.csv"
    path.write_bytes(data)

    try:
        df = read_table(path)
    except ValueError:
        return  # an honest refusal on an unreadable file is acceptable
    # The only acceptable outcomes are a parsed table or a ValueError; any other
    # exception escaping read_table would be a loader bug.
    assert df.shape[1] >= 1
