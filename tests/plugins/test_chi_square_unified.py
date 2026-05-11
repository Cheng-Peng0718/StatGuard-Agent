import pandas as pd

from core.analysis_tool_plugins import get_plugin


class DummyContext:
    def __init__(self, df, args=None):
        self.df = df
        self.args = args or {}

    def load_df(self):
        return self.df

    def get_arg(self, key, default=None):
        return self.args.get(key, default)


def test_chi_square_unified_execute_and_analysis_run():
    plugin = get_plugin("run_chi_square")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "sex": ["M", "M", "M", "F", "F", "F", "M", "F"],
        "smoke": ["Y", "Y", "N", "Y", "N", "N", "N", "Y"],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "row_col": "sex",
            "col_col": "smoke",
        },
    ))

    assert raw["status"] in {"ok", "warning"}
    assert raw["details"]["method"] == "Chi-square test of independence"
    assert raw["details"]["nobs"] == 8
    assert "chi_square_statistic" in raw["details"]
    assert "p_value" in raw["details"]
    assert "observed_table" in raw["details"]
    assert "expected_table" in raw["details"]

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "row_col": "sex",
            "col_col": "smoke",
        },
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "run_chi_square"
    assert run["title"] == "Chi-square Test: sex × smoke"

    assert "chi_square_statistic" in run["metrics"]
    assert "p_value" in run["metrics"]
    assert "observed_table" in run["tables"]

    # Expected table should stay metadata by default.
    assert "expected_table" not in run["tables"]
    assert "expected_table" in run["metadata"]

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Chi-square statistic" in labels
    assert "p-value" in labels
    assert "Expected cells below 5" in labels

    table_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "table"
    )

    table_labels = [col["label"] for col in table_block["columns"]]

    assert "Row level" in table_labels
    assert "Y" in table_labels or "N" in table_labels


def test_chi_square_blocks_missing_column():
    plugin = get_plugin("run_chi_square")

    df = pd.DataFrame({
        "sex": ["M", "F"],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "row_col": "sex",
            "col_col": "smoke",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "COLUMNS_NOT_FOUND"


def test_chi_square_blocks_insufficient_levels():
    plugin = get_plugin("run_chi_square")

    df = pd.DataFrame({
        "sex": ["M", "M", "M"],
        "smoke": ["Y", "N", "Y"],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "row_col": "sex",
            "col_col": "smoke",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "INSUFFICIENT_LEVELS"