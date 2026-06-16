import tempfile

import pandas as pd

from core.analysis_tool_plugins import get_plugin
from core.data_versions import create_child_data_version
from core.schema import AgentContext


def _seed(df, args):
    """A real context seeded with an initial data version, so clean_data's
    create_child_data_version + active-version switch run for real."""
    ws = tempfile.mkdtemp()
    v0 = create_child_data_version(df, ws, parent_version_id=None,
                                   operation="initial_load", created_by="test")
    return AgentContext(workspace_dir=ws, arguments=args,
                        data_versions=[v0], active_data_version_id=v0["version_id"])


def _cleaned_df(raw):
    """Read the cleaned frame back from the new active version (the real handoff)."""
    return pd.read_parquet(raw["data_version_update"]["new_version"]["path"])


def test_clean_data_drop_rows_creates_data_version_update():
    plugin = get_plugin("clean_data")

    assert plugin is not None
    assert plugin.execute is not None
    # P3: clean_data is no longer a blanket-confirmation tool. Gating is per-action
    # via confirmation_policy (only destructive actions need review).
    assert plugin.requires_confirmation is False
    assert plugin.confirmation_policy is not None

    df = pd.DataFrame({
        "GPA": [3.0, None, 4.0],
        "SATM": [600, 700, None],
        "Other": ["a", "b", "c"],
    })

    ctx = _seed(
        df,
        {
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
    )

    raw = plugin.run(ctx)

    assert raw["status"] == "ok"
    out = _cleaned_df(raw)
    assert len(out) == 1

    assert raw["details"]["original_n_rows"] == 3
    assert raw["details"]["final_n_rows"] == 1
    assert raw["details"]["rows_removed"] == 2
    assert raw["details"]["data_version_created"] is True

    assert "data_version_update" in raw
    assert "data_version_update" in raw["details"]

    update = raw["data_version_update"]
    assert update["old_version_id"] == ctx.active_data_version_id
    assert update["new_version_id"].startswith("data_v_")
    assert update["operation"] == "clean_data"

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "clean_data"
    assert run["title"] == "Data Cleaning"
    assert run["metrics"]["original_n_rows"] == 3
    assert run["metrics"]["final_n_rows"] == 1
    assert run["metrics"]["rows_removed"] == 2
    assert "data_version_update" in run["metadata"]
    assert run["report_blocks"]


def test_clean_data_blocks_missing_column():
    plugin = get_plugin("clean_data")

    df = pd.DataFrame({
        "GPA": [3.0, 4.0],
    })

    ctx = _seed(
        df,
        {
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["SATM"],
        },
    )

    raw = plugin.run(ctx)

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "COLUMNS_NOT_FOUND"


def test_clean_data_impute_mean():
    plugin = get_plugin("clean_data")

    df = pd.DataFrame({
        "x": [1.0, None, 3.0],
    })

    ctx = _seed(
        df,
        {
            "action_type": "impute",
            "strategy": "mean",
            "columns": ["x"],
        },
    )

    raw = plugin.run(ctx)

    assert raw["status"] == "ok"
    out = _cleaned_df(raw)
    assert out["x"].isna().sum() == 0
    assert out.loc[1, "x"] == 2.0