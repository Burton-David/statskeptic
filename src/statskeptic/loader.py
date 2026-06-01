"""Read a CSV into a DataFrame, robustly.

Real-world CSVs are not all comma-separated UTF-8 with one clean header row. CleverCSV
detects the dialect (delimiter, quote character, encoding) from the file's own row and
type patterns, so a semicolon-delimited or Latin-1 file loads as the table it actually
is instead of a single mangled column. pandas is the fallback for the rare file CleverCSV
trips on, and a file that neither can read is a usage error, not a silent empty frame.
"""

from __future__ import annotations

from pathlib import Path

import clevercsv
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    src = str(path)
    df: pd.DataFrame | None = None
    try:
        df = clevercsv.read_dataframe(src)
    except Exception:
        # CleverCSV is the more robust reader, so if it fails the file is likely broken;
        # still give pandas a turn before declaring the file unreadable.
        try:
            df = pd.read_csv(src)
        except Exception as exc:
            raise ValueError(f"could not read {src!r} as a CSV: {exc}") from exc

    if df is None or df.shape[1] == 0:
        raise ValueError(f"{src!r} did not parse into any columns")
    return df
