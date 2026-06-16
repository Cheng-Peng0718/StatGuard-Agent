"""
Regression tests for the extended clean_data actions (P2).

Per the data-prep design: the clean tools are self-built deterministic Python;
pandera is the INDEPENDENT oracle in the test harness -- it asserts the cleaned
frame conforms to an expected schema (no nulls after impute, no duplicate rows,
only canonical category labels, correct post-cast constraints). Two independent
implementations cross-checking each other, the same way scipy validates the
stats plugins.
"""

import os  # noqa: F401
import tempfile

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

import core.analysis_tool_plugins  # noqa: F401
from core.analysis_tool_plugins.registry import get_plugin
from core.data_versions import create_child_data_version
from core.schema import AgentContext


def _run(df, **args):
    """Run clean_data through the REAL version mechanism: seed an initial data
    version, execute, then read the cleaned frame back from the new active
    version's parquet (the same handoff the graph performs in production)."""
    plugin = get_plugin("clean_data")
    ws = tempfile.mkdtemp()
    v0 = create_child_data_version(df, ws, parent_version_id=None,
                                   operation="initial_load", created_by="test",
                                   description="raw")
    ctx = AgentContext(workspace_dir=ws, arguments=args,
                       data_versions=[v0], active_data_version_id=v0["version_id"])
    result = plugin.execute(ctx)
    assert result["status"] == "ok", result
    upd = result["data_version_update"]
    # the graph switches the active version to this; read the cleaned frame from it
    assert upd.get("active_data_version_id") == upd["new_version"]["version_id"]
    out = pd.read_parquet(upd["new_version"]["path"])
    return out, result["details"]


# --------------------------------------------------------------------------
# cast
# --------------------------------------------------------------------------
def test_cast_numeric_from_text():
    df = pd.DataFrame({"rev": ["1,234", "$5,000", "2,100", "900"]})
    out, _ = _run(df, action_type="cast", strategy="numeric", columns=["rev"])
    assert pd.api.types.is_numeric_dtype(out["rev"])
    assert out["rev"].tolist() == [1234, 5000, 2100, 900]
    # pandera oracle: a positive, non-null numeric column
    pa.DataFrameSchema({"rev": pa.Column(checks=pa.Check.gt(0), nullable=False)}).validate(out)


def test_cast_datetime_from_text():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-02-15", "2023-12-09"]})
    out, _ = _run(df, action_type="cast", strategy="datetime", columns=["d"])
    assert pd.api.types.is_datetime64_any_dtype(out["d"])
    assert out["d"].isna().sum() == 0


def test_cast_counts_coercion_failures():
    df = pd.DataFrame({"x": ["1", "2", "not_a_number", "4"]})
    out, det = _run(df, action_type="cast", strategy="numeric", columns=["x"])
    assert det["action_info"]["cast"]["x"]["coercion_failures"] == 1
    assert out["x"].isna().sum() == 1


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------
def test_dedup_full_rows():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 2], "b": ["x", "x", "y", "y", "z"]})
    out, det = _run(df, action_type="dedup", strategy="rows")
    assert det["action_info"]["rows_removed"] == 2
    assert out.duplicated().sum() == 0
    # pandera oracle: the (a,b) pair is unique
    pa.DataFrameSchema(unique=["a", "b"]).validate(out)


def test_dedup_on_subset():
    df = pd.DataFrame({"id": [1, 1, 2], "v": [10, 99, 20]})
    out, det = _run(df, action_type="dedup", strategy="rows", columns=["id"])
    assert len(out) == 2
    pa.DataFrameSchema(unique=["id"]).validate(out)


# --------------------------------------------------------------------------
# standardize categories
# --------------------------------------------------------------------------
def test_standardize_categories_collapses_case_and_whitespace():
    df = pd.DataFrame({"ch": [" Online", "online", "ONLINE", "Email", "email", "Email "]})
    out, det = _run(df, action_type="standardize", strategy="categories", columns=["ch"])
    assert out["ch"].nunique() == 2
    info = det["action_info"]["standardize"]["ch"]
    assert info["labels_before"] > info["labels_after"]
    # pandera oracle: only the two canonical labels survive
    canon = set(out["ch"].unique())
    pa.DataFrameSchema({"ch": pa.Column(checks=pa.Check.isin(canon))}).validate(out)


# --------------------------------------------------------------------------
# impute mode / constant
# --------------------------------------------------------------------------
def test_impute_mode_categorical():
    df = pd.DataFrame({"seg": ["A", "A", None, "B", None]})
    out, _ = _run(df, action_type="impute", strategy="mode", columns=["seg"])
    assert out["seg"].isna().sum() == 0
    assert out["seg"].tolist() == ["A", "A", "A", "B", "A"]   # mode = "A"
    pa.DataFrameSchema({"seg": pa.Column(str, nullable=False)}).validate(out)


def test_impute_constant_requires_fill_value():
    df = pd.DataFrame({"seg": ["A", None]})
    plugin = get_plugin("clean_data")
    ws = tempfile.mkdtemp()
    v0 = create_child_data_version(df, ws, parent_version_id=None,
                                   operation="initial_load", created_by="test")
    ctx = AgentContext(workspace_dir=ws,
                       arguments={"action_type": "impute", "strategy": "constant",
                                  "columns": ["seg"]},
                       data_versions=[v0], active_data_version_id=v0["version_id"])
    result = plugin.execute(ctx)
    assert result["status"] == "blocked"
    assert result["error_code"] == "MISSING_FILL_VALUE"


def test_impute_constant_fills():
    df = pd.DataFrame({"seg": ["A", None, "B"]})
    out, _ = _run(df, action_type="impute", strategy="constant",
                  columns=["seg"], fill_value="UNKNOWN")
    assert out["seg"].tolist() == ["A", "UNKNOWN", "B"]
    pa.DataFrameSchema({"seg": pa.Column(str, nullable=False)}).validate(out)


# --------------------------------------------------------------------------
# backward compatibility: the original actions still work
# --------------------------------------------------------------------------
def test_legacy_drop_and_impute_still_work():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, 2.0, 3.0]})
    out_drop, _ = _run(df, action_type="drop", strategy="rows")
    assert out_drop["x"].isna().sum() == 0 and out_drop["y"].isna().sum() == 0
    out_mean, _ = _run(df, action_type="impute", strategy="mean", columns=["x"])
    assert out_mean["x"].isna().sum() == 0


# --------------------------------------------------------------------------
# clip outliers + ffill/bfill (P2 remainder)
# --------------------------------------------------------------------------
def test_clip_outliers_winsorises_without_dropping_rows():
    df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15, 16, 1000]})  # 1000 is far-out
    out, det = _run(df, action_type="clip", strategy="outliers", columns=["x"])
    assert len(out) == len(df)                 # no rows removed
    assert out["x"].max() < 1000               # extreme value pulled to the fence
    assert det["action_info"]["clip"]["x"]["n_clipped"] == 1
    upper = det["action_info"]["clip"]["x"]["upper"]
    pa.DataFrameSchema({"x": pa.Column(checks=pa.Check.le(upper))}).validate(out)


def test_impute_ffill():
    df = pd.DataFrame({"v": [1.0, np.nan, np.nan, 4.0]})
    out, _ = _run(df, action_type="impute", strategy="ffill", columns=["v"])
    assert out["v"].tolist() == [1.0, 1.0, 1.0, 4.0]


def test_impute_bfill():
    df = pd.DataFrame({"v": [1.0, np.nan, np.nan, 4.0]})
    out, _ = _run(df, action_type="impute", strategy="bfill", columns=["v"])
    assert out["v"].tolist() == [1.0, 4.0, 4.0, 4.0]