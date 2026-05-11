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


def test_regression_diagnostics_unified_execute_and_analysis_run():
    plugin = get_plugin("regression_diagnostics")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "x": [2, 4, 6, 8, 10, 12, 14, 16],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "y",
            "feature_cols": ["x"],
            "min_n_per_parameter": 1,
        },
    ))

    assert raw["status"] in {"ok", "warning"}
    assert "vif" in raw["details"]
    assert "breusch_pagan" in raw["details"]
    assert raw["details"]["p_eff"] == 1

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "target_col": "y",
            "feature_cols": ["x"],
            "min_n_per_parameter": 1,
        },
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=raw.get("artifacts", []),
        observation_id="obs_test",
    )

    assert run["tool_name"] == "regression_diagnostics"
    assert run["title"] == "Model Diagnostics"

    assert "max_vif" in run["metrics"]
    assert "breusch_pagan_lm_p_value" in run["metrics"]
    assert "heteroscedasticity_flag_0_05" in run["metrics"]

    assert "vif" in run["tables"]

    # Raw BP dict should be metadata, not a report table.
    assert "breusch_pagan" not in run["tables"]
    assert "breusch_pagan" in run["metadata"]

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Maximum VIF" in labels
    assert "Breusch-Pagan LM p-value" in labels
    assert "Heteroscedasticity flag" in labels

    table_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "table"
    )

    table_labels = [col["label"] for col in table_block["columns"]]

    assert "Term" in table_labels
    assert "VIF" in table_labels
    assert "High VIF flag" in table_labels


def test_regression_diagnostics_blocks_missing_feature_column():
    plugin = get_plugin("regression_diagnostics")

    df = pd.DataFrame({
        "y": [1, 2, 3],
        "x": [1, 2, 3],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "y",
            "feature_cols": ["z"],
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "COLUMNS_NOT_FOUND"