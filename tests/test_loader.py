"""The CSV reader must recover the table from messy real-world dialects, not just clean
comma-separated UTF-8. These files would each collapse to a single column under
pd.read_csv defaults; CleverCSV detects the dialect and reads them as intended."""

from __future__ import annotations

import numpy as np
import pytest

from statskeptic import Verdict, analyze
from statskeptic.loader import read_table


def _rows() -> list[tuple[str, float]]:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20)
    b = rng.normal(1.2, 1, 20)
    return [("ctrl", float(v)) for v in a] + [("drug", float(v)) for v in b]


def _write(path, header_sep, sep, encoding="utf-8", bom=False):
    lines = [header_sep.join(["arm", "score"])]
    lines += [f"{g}{sep}{v:.4f}" for g, v in _rows()]
    text = "\n".join(lines) + "\n"
    data = text.encode(encoding)
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    "sep",
    [";", "\t", "|"],
    ids=["semicolon", "tab", "pipe"],
)
def test_reader_recovers_nonstandard_delimiters(tmp_path, sep):
    path = _write(tmp_path / "d.csv", sep, sep)
    df = read_table(path)
    assert list(df.columns) == ["arm", "score"]
    assert df.shape == (40, 2)
    # The whole pipeline runs on the recovered table.
    report = analyze(path, "Does the drug change the score?")
    assert report.verdict in set(Verdict)
    assert report.analyses and report.analyses[0].result.n == 40


def test_reader_handles_utf8_bom(tmp_path):
    path = _write(tmp_path / "bom.csv", ",", ",", bom=True)
    df = read_table(path)
    # The BOM must not contaminate the first column name.
    assert list(df.columns) == ["arm", "score"]
    assert df.shape == (40, 2)


def test_reader_handles_latin1_encoding(tmp_path):
    path = tmp_path / "latin1.csv"
    rows = "\n".join(f"{g},{v:.3f}" for g, v in _rows())
    # A non-ASCII byte (0xe9 = 'é' in Latin-1) in a value, written as Latin-1.
    text = "r\xe9gion,score\n" + rows + "\n"
    path.write_bytes(text.encode("latin-1"))
    df = read_table(path)
    assert df.shape[1] == 2 and df.shape[0] == 40


def test_unreadable_file_is_a_value_error(tmp_path):
    path = tmp_path / "junk.bin"
    path.write_bytes(b"\x00\x01\x02 not a table at all \xff\xfe")
    with pytest.raises(ValueError):
        read_table(path)
