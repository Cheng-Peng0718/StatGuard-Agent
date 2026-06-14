"""Regression tests for the data_quality_report diagnosis plugin."""

import numpy as np
import pandas as pd
import pytest

import core.analysis_tool_plugins  # noqa: F401  (triggers load_plugins)
from core.analysis_tool_plugins.registry import get_plugin


class _Ctx:
    def __init__(self, df):
        self._df = df

    def load_df(self):
        return self._df

    def get_arg(self, name, default=None):
        return default


def _run(df):
    plugin = get_plugin("data_quality_report")
    assert plugin is not None, "data_quality_report not auto-registered"
    result = plugin.execute(_Ctx(df))
    assert result["status"] == "ok", result
    return result["details"]


@pytest.fixture
def messy_df():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        "revenue_text": ["1,234", "$5,000", "2,100", "900", "$3,300"] * 8,
        "signup_date": ["2024-01-01", "2024-02-15", "2023-12-09",
                        "2024-03-03", "2024-01-20"] * 8,
        "channel": [" Online", "online", "ONLINE", "Email", "email"] * 8,
        "country": ["USA", "NA", "USA", "?", "USA"] * 8,
        "plan": ["pro"] * 40,
        "amount": np.r_[rng.normal(100, 10, 39), 99999.0],
        "user_ref": [f"u{i}" for i in range(40)],
        "mixed": [1, "two", 3, "four", 5] * 8,
        "clean_num": rng.normal(50, 5, 40).round(2),
    })
    miss = base.copy()
    miss.loc[:9, "clean_num"] = np.nan          # 25% missing -> high severity
    return pd.concat([miss, miss.iloc[:3]], ignore_index=True)  # + duplicate rows


def test_detects_every_issue_type(messy_df):
    details = _run(messy_df)
    found = {(i["column"], i["issue"]) for i in details["issues"]}
    expected = {
        ("(table)", "duplicate_rows"),
        ("revenue_text", "numeric_stored_as_text"),
        ("signup_date", "date_stored_as_text"),
        ("channel", "inconsistent_categories"),
        ("country", "disguised_missing"),
        ("plan", "constant_column"),
        ("amount", "outliers"),
        ("user_ref", "high_cardinality"),
        ("mixed", "mixed_types"),
        ("clean_num", "missing_values"),
    }
    assert expected <= found, f"missed: {expected - found}"


def test_severity_and_shape(messy_df):
    details = _run(messy_df)
    assert details["shape"]["rows"] == 43
    assert details["shape"]["columns"] == 9
    assert details["duplicate_rows"] == 3
    sev = details["severity_counts"]
    assert sev["high"] >= 3        # constant + high-missing + dup rows
    assert details["n_issues"] == sum(sev.values())


def test_single_far_out_value_flags_outlier():
    df = pd.DataFrame({"x": list(np.random.default_rng(1).normal(100, 5, 200)) + [50000.0]})
    details = _run(df)
    assert any(i["issue"] == "outliers" for i in details["issues"])


def test_clean_data_has_no_false_positives():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, 200),
        "income": rng.normal(60000, 8000, 200).round(2),
        "segment": rng.choice(["Consumer", "Corporate", "Small Business"], 200),
        "region": rng.choice(["North", "South", "East", "West"], 200),
    })
    details = _run(df)
    # a genuinely clean frame should raise no issues
    assert details["n_issues"] == 0, [i["issue"] for i in details["issues"]]


def test_column_diagnostics_present(messy_df):
    details = _run(messy_df)
    diag = {d["column"]: d for d in details["column_diagnostics"]}
    assert diag["plan"]["role"] == "constant"
    assert diag["user_ref"]["role"] in ("id_like", "text")
    assert diag["amount"]["role"] == "numeric"