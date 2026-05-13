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


def test_linear_model_unified_execute_and_analysis_run():
    plugin = get_plugin("run_multiple_regression")

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
    assert "r_squared" in raw["details"]
    assert "coef_table" in raw["details"]
    assert "coefficient_interpretations" in raw["details"]
    assert "assumptions_and_limitations" in raw["details"]
    assert "significant_predictor_count" in raw["details"]
    assert "model_significant_at_alpha" in raw["details"]
    assert raw["details"]["nobs"] == 8
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

    assert run["tool_name"] == "run_multiple_regression"
    assert run["title"] == "Linear Model: y ~ x"

    assert "nobs" in run["metrics"]
    assert "r_squared" in run["metrics"]
    assert "coef_table" in run["tables"]
    assert "coefficient_interpretations" in run["tables"]
    assert "assumptions_and_limitations" in run["tables"]
    assert "significant_predictor_count" in run["metrics"]
    assert "model_significant_at_alpha" in run["metrics"]

    # Technical fields should be metadata, not user-facing metrics.
    assert "aic" not in run["metrics"]
    assert "bic" not in run["metrics"]
    assert "p_eff" not in run["metrics"]

    assert "aic" in run["metadata"]
    assert "bic" in run["metadata"]
    assert "p_eff" in run["metadata"]

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Observations used" in labels
    assert "R-squared" in labels
    assert "Adjusted R-squared" in labels
    assert "Model p-value" in labels
    assert "Overall model significant" in labels
    assert "Significant non-intercept predictors" in labels

    table_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "table"
    )

    table_titles = [
        block.get("title")
        for block in run["report_blocks"]
        if block["type"] == "table"
    ]

    assert "Coefficient interpretations" in table_titles
    assert "Assumptions and limitations" in table_titles

    table_labels = [col["label"] for col in table_block["columns"]]
    first_term_value = table_block["rows"][0][0]

    assert "Estimate" in table_labels
    assert "p-value" in table_labels
    assert first_term_value == "Intercept"


def test_linear_model_blocks_missing_feature_column():
    plugin = get_plugin("run_multiple_regression")

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