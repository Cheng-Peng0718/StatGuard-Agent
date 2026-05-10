import pandas as pd
import pytest

from core.data.tabular_io import load_tabular_dataframe


def test_load_tabular_dataframe_reads_csv(tmp_path):
    path = tmp_path / "student_data.csv"

    pd.DataFrame({
        "GPA": [3.0, 3.5],
        "SATM": [600, 700],
    }).to_csv(path, index=False)

    df = load_tabular_dataframe(path)

    assert list(df.columns) == ["GPA", "SATM"]
    assert df.shape == (2, 2)


def test_load_tabular_dataframe_reads_xlsx(tmp_path):
    path = tmp_path / "student_data.xlsx"

    pd.DataFrame({
        "GPA": [3.0, 3.5],
        "SATM": [600, 700],
    }).to_excel(path, index=False)

    df = load_tabular_dataframe(path)

    assert list(df.columns) == ["GPA", "SATM"]
    assert df.shape == (2, 2)


def test_load_tabular_dataframe_uses_xlrd_for_xls(monkeypatch, tmp_path):
    path = tmp_path / "student_data.xls"
    path.write_bytes(b"fake xls content")

    seen = {}

    def fake_read_excel(received_path, engine=None):
        seen["path"] = received_path
        seen["engine"] = engine
        return pd.DataFrame({"GPA": [3.0]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    df = load_tabular_dataframe(path)

    assert seen["engine"] == "xlrd"
    assert list(df.columns) == ["GPA"]


def test_load_tabular_dataframe_rejects_unknown_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a table", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported tabular file type"):
        load_tabular_dataframe(path)