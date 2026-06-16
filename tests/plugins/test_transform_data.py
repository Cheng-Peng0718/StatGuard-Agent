"""P4 tests for transform_data: structured derive/filter/bin/encode/rename.

Self-built deterministic transforms; pandera is the INDEPENDENT oracle that the
transformed frame conforms to the expected schema (new column present and typed,
filtered rows satisfy the predicate, one-hot columns are 0/1, etc.).
"""
import tempfile

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

import core.analysis_tool_plugins  # noqa: F401
from core.analysis_tool_plugins.registry import get_plugin
from core.analysis_tool_plugins.plugins.transform_data import (
    evaluate_transform_data_guardrails,
)
from core.data_versions import create_child_data_version
from core.schema import AgentContext


def _ctx(df, **args):
    ws = tempfile.mkdtemp()
    v0 = create_child_data_version(df, ws, parent_version_id=None,
                                   operation="initial_load", created_by="test")
    return AgentContext(workspace_dir=ws, arguments=args,
                        data_versions=[v0], active_data_version_id=v0["version_id"])


def _run(df, **args):
    plugin = get_plugin("transform_data")
    result = plugin.execute(_ctx(df, **args))
    assert result["status"] == "ok", result
    upd = result["data_version_update"]
    assert upd.get("active_data_version_id") == upd["new_version"]["version_id"]
    out = pd.read_parquet(upd["new_version"]["path"])
    return out, result["details"]


def _blocked(df, **args):
    plugin = get_plugin("transform_data")
    return plugin.execute(_ctx(df, **args))


# --------------------------------------------------------------------------
# derive
# --------------------------------------------------------------------------
def test_derive_column_minus_column():
    df = pd.DataFrame({"revenue": [100.0, 200.0], "cost": [60.0, 150.0]})
    out, det = _run(df, action_type="derive", new_column="margin",
                    left="revenue", op="-", right="cost")
    assert out["margin"].tolist() == [40.0, 50.0]
    assert det["cols_added"] == 1
    pa.DataFrameSchema({"margin": pa.Column(float, nullable=False)}).validate(out)


def test_derive_column_times_constant():
    df = pd.DataFrame({"price": [10.0, 20.0]})
    out, _ = _run(df, action_type="derive", new_column="marked_up",
                  left="price", op="*", right=1.2)
    assert out["marked_up"].tolist() == [12.0, 24.0]


def test_derive_division_tracks_inf():
    df = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 0.0]})
    out, det = _run(df, action_type="derive", new_column="ratio",
                    left="a", op="/", right="b")
    assert det["action_info"]["n_inf"] == 1


def test_derive_bad_operand_blocked():
    df = pd.DataFrame({"a": [1.0]})
    res = _blocked(df, action_type="derive", new_column="x",
                   left="a", op="+", right="not_a_col_or_number")
    assert res["status"] == "blocked" and res["error_code"] == "BAD_OPERAND"


# --------------------------------------------------------------------------
# filter
# --------------------------------------------------------------------------
def test_filter_and_conditions():
    df = pd.DataFrame({"region": ["W", "E", "W", "W"], "rev": [10, 50, 200, 30]})
    out, det = _run(df, action_type="filter",
                    conditions=[{"column": "region", "op": "==", "value": "W"},
                                {"column": "rev", "op": ">", "value": 20}])
    assert det["action_info"]["rows_kept"] == 2
    # pandera oracle: every surviving row satisfies the predicate
    pa.DataFrameSchema({
        "region": pa.Column(checks=pa.Check.eq("W")),
        "rev": pa.Column(checks=pa.Check.gt(20)),
    }).validate(out)


def test_filter_between_and_in():
    df = pd.DataFrame({"x": [1, 5, 9, 12], "g": ["a", "b", "c", "a"]})
    out, _ = _run(df, action_type="filter",
                  conditions=[{"column": "x", "op": "between", "value": [4, 10]},
                              {"column": "g", "op": "in", "value": ["b", "c"]}])
    assert out["x"].tolist() == [5, 9]


# --------------------------------------------------------------------------
# bin
# --------------------------------------------------------------------------
def test_bin_quantile():
    df = pd.DataFrame({"score": list(range(100))})
    out, det = _run(df, action_type="bin", column="score", new_column="quartile",
                    method="quantile", n_bins=4)
    assert "quartile" in out.columns
    assert out["quartile"].dropna().nunique() == 4


def test_bin_explicit_edges_with_labels():
    df = pd.DataFrame({"age": [5, 17, 35, 70]})
    out, _ = _run(df, action_type="bin", column="age", new_column="band",
                  edges=[0, 18, 65, 120], labels=["minor", "adult", "senior"])
    assert out["band"].tolist() == ["minor", "minor", "adult", "senior"]
    pa.DataFrameSchema({"band": pa.Column(checks=pa.Check.isin(
        ["minor", "adult", "senior"]))}).validate(out)


def test_bin_non_numeric_blocked():
    df = pd.DataFrame({"c": ["a", "b"]})
    res = _blocked(df, action_type="bin", column="c")
    assert res["status"] == "blocked" and res["error_code"] == "NOT_NUMERIC"


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------
def test_encode_onehot():
    df = pd.DataFrame({"color": ["red", "blue", "red"], "v": [1, 2, 3]})
    out, det = _run(df, action_type="encode", column="color", method="onehot")
    assert "color" not in out.columns                 # original dropped
    assert det["action_info"]["n_new_columns"] == 2
    dummy_cols = det["action_info"]["new_columns"]
    schema = pa.DataFrameSchema({c: pa.Column(int, pa.Check.isin([0, 1])) for c in dummy_cols})
    schema.validate(out)


def test_encode_ordinal_respects_order():
    df = pd.DataFrame({"size": ["S", "L", "M"]})
    out, _ = _run(df, action_type="encode", column="size", method="ordinal",
                  order=["S", "M", "L"])
    assert out["size_ordinal"].tolist() == [0, 2, 1]


def test_encode_label():
    df = pd.DataFrame({"c": ["x", "y", "x"]})
    out, det = _run(df, action_type="encode", column="c", method="label")
    assert det["action_info"]["n_labels"] == 2
    assert out["c_code"].tolist() == [0, 1, 0]


# --------------------------------------------------------------------------
# rename
# --------------------------------------------------------------------------
def test_rename_mapping():
    df = pd.DataFrame({"old_a": [1], "b": [2]})
    out, det = _run(df, action_type="rename", mapping={"old_a": "new_a"})
    assert "new_a" in out.columns and "old_a" not in out.columns
    assert det["action_info"]["renamed"] == ["old_a"]


def test_rename_missing_column_blocked():
    df = pd.DataFrame({"a": [1]})
    res = _blocked(df, action_type="rename", mapping={"nope": "x"})
    assert res["status"] == "blocked" and res["error_code"] == "COLUMN_NOT_FOUND"


# --------------------------------------------------------------------------
# unsupported
# --------------------------------------------------------------------------
def test_unsupported_action_blocked():
    df = pd.DataFrame({"a": [1]})
    res = _blocked(df, action_type="pivot")
    assert res["status"] == "blocked" and res["error_code"] == "UNSUPPORTED_TRANSFORM_ACTION"


# --------------------------------------------------------------------------
# guardrails (post-exec transparency)
# --------------------------------------------------------------------------
def test_guardrail_filter_large_removal_warns():
    f = evaluate_transform_data_guardrails({
        "metrics": {"original_n_rows": 100, "rows_removed": 80},
        "metadata": {"action_type": "filter"}})
    assert any(x["severity"] == "warning" for x in f)


def test_guardrail_onehot_explosion_warns():
    f = evaluate_transform_data_guardrails({
        "metrics": {}, "metadata": {"action_type": "encode",
                                     "action_info": {"method": "onehot",
                                                     "column": "city", "n_new_columns": 40}}})
    assert any(x["severity"] == "warning" for x in f)


def test_guardrail_derive_inf_warns():
    f = evaluate_transform_data_guardrails({
        "metrics": {}, "metadata": {"action_type": "derive",
                                     "action_info": {"new_column": "ratio", "n_inf": 3}}})
    assert any(x["severity"] == "warning" for x in f)


def test_guardrail_clean_transform_silent():
    f = evaluate_transform_data_guardrails({
        "metrics": {"original_n_rows": 100, "rows_removed": 0},
        "metadata": {"action_type": "rename", "action_info": {}}})
    assert f == []


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------
def test_plugin_wired():
    p = get_plugin("transform_data")
    assert p.requires_confirmation is False
    assert p.guardrail_evaluators