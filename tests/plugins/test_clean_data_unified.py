import pandas as pd

from core.analysis_tool_plugins import get_plugin


class DummyContext:
    def __init__(self, df, args=None):
        self.df = df.copy()
        self.saved_df = None
        self.args = args or {}
        self.workspace_dir = "."
        self.active_data_version_id = "raw_v1"
        self.data_versions = []
        self.data_audit_log = []

    def load_df(self):
        return self.df

    def save_df(self, df):
        self.saved_df = df.copy()
        self.df = df.copy()

    def get_arg(self, key, default=None):
        return self.args.get(key, default)


def test_clean_data_drop_rows_creates_data_version_update():
    plugin = get_plugin("clean_data")

    assert plugin is not None
    assert plugin.execute is not None
    assert plugin.requires_confirmation is True

    df = pd.DataFrame({
        "GPA": [3.0, None, 4.0],
        "SATM": [600, 700, None],
        "Other": ["a", "b", "c"],
    })

    ctx = DummyContext(
        df,
        {
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
    )

    raw = plugin.run(ctx)

    assert raw["status"] == "ok"
    assert ctx.saved_df is not None
    assert len(ctx.saved_df) == 1

    assert raw["details"]["original_n_rows"] == 3
    assert raw["details"]["final_n_rows"] == 1
    assert raw["details"]["rows_removed"] == 2
    assert raw["details"]["data_version_created"] is True

    assert "data_version_update" in raw
    assert "data_version_update" in raw["details"]

    update = raw["data_version_update"]
    assert update["old_version_id"] == "raw_v1"
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

    ctx = DummyContext(
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

    ctx = DummyContext(
        df,
        {
            "action_type": "impute",
            "strategy": "mean",
            "columns": ["x"],
        },
    )

    raw = plugin.run(ctx)

    assert raw["status"] == "ok"
    assert ctx.saved_df is not None
    assert ctx.saved_df["x"].isna().sum() == 0
    assert ctx.saved_df.loc[1, "x"] == 2.0