from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd


SUPPORTED_TABULAR_EXTENSIONS: Set[str] = {
    ".csv",
    ".xlsx",
    ".xls",
    ".parquet",
}


def load_tabular_dataframe(path: str | Path) -> pd.DataFrame:
    """
    Load a supported tabular dataset file into a DataFrame.

    Supported formats:
    - .csv
    - .xlsx
    - .xls
    - .parquet

    This is the single backend reader used by upload and dataset context refresh.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_TABULAR_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_TABULAR_EXTENSIONS))
        raise ValueError(
            f"Unsupported tabular file type '{suffix}'. "
            f"Supported types: {supported}."
        )

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".xlsx":
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise RuntimeError(
                "Reading .xlsx files requires the optional dependency "
                "`openpyxl`. Install it with: pip install openpyxl"
            ) from exc

    if suffix == ".xls":
        try:
            return pd.read_excel(path, engine="xlrd")
        except ImportError as exc:
            raise RuntimeError(
                "Reading .xls files requires the optional dependency "
                "`xlrd`. Install it with: pip install xlrd"
            ) from exc

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported tabular file type '{suffix}'.")