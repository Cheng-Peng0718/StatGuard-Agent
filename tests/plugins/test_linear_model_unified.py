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

def test_linear_model_interprets_categorical_dummy_terms_relative_to_reference(tmp_path):
    import pandas as pd

    from core.analysis_tool_plugins import get_plugin

    class DummyContext:
        def __init__(self, df, args):
            self.df = df
            self.arguments = args
            self.args = args

        def load_df(self):
            return self.df

        def get_arg(self, name, default=None):
            return self.arguments.get(name, default)

    df = pd.DataFrame({
        "total_revenue": [
            100, 120, 130, 150,
            180, 200, 210, 220,
            90, 95, 105, 110,
            160, 170, 175, 185,
        ],
        "number_of_orders": [
            1, 2, 2, 3,
            3, 4, 4, 5,
            1, 1, 2, 2,
            2, 3, 3, 4,
        ],
        "region": [
            "East", "East", "East", "East",
            "North", "North", "North", "North",
            "South", "South", "South", "South",
            "West", "West", "West", "West",
        ],
    })

    plugin = get_plugin("run_multiple_regression")
    assert plugin is not None

    raw = plugin.run(
        DummyContext(
            df,
            {
                "target_col": "total_revenue",
                "feature_cols": ["number_of_orders", "region"],
            },
        )
    )

    assert raw["status"] in {"ok", "warning"}

    interpretations = raw["details"]["coefficient_interpretations"]

    north_row = next(
        row for row in interpretations
        if row["term"] == "region_North"
    )

    assert north_row["variable_type"] == "categorical_dummy"
    assert north_row["source_feature"] == "region"
    assert north_row["level"] == "North"
    assert north_row["reference_level"] == "East"
    assert "Compared with the reference category" in north_row["interpretation"]
    assert "`region = East`" in north_row["interpretation"]
    assert "`region = North`" in north_row["interpretation"]
    assert "higher `region_North`" not in north_row["interpretation"]

    numeric_row = next(
        row for row in interpretations
        if row["term"] == "number_of_orders"
    )

    assert numeric_row["variable_type"] == "numeric_or_continuous"
    assert "For a one-unit increase" in numeric_row["interpretation"]

def test_linear_model_hc3_summary_and_metrics_are_consistent():
    import pandas as pd

    plugin = get_plugin("run_multiple_regression")
    assert plugin is not None

    class LocalContext:
        def __init__(self, df, args):
            self.df = df
            self.args = args
            self.arguments = args
            self.active_data_version_id = "data_v_test"

        def load_df(self):
            return self.df.copy()

        def get_arg(self, name, default=None):
            return self.args.get(name, default)

    n = 80

    df = pd.DataFrame({
        "x": list(range(1, n + 1)),
        "z": [i % 5 for i in range(n)],
        "segment": [
            "A" if i % 3 == 0 else "B" if i % 3 == 1 else "C"
            for i in range(n)
        ],
    })

    # Deterministic heteroscedastic-ish outcome.
    df["y"] = [
        100
        + 3.0 * df.loc[i, "x"]
        + 1.5 * df.loc[i, "z"]
        + (8 if df.loc[i, "segment"] == "B" else 15 if df.loc[i, "segment"] == "C" else 0)
        + ((i % 7) - 3) * (1 + df.loc[i, "x"] / 25)
        for i in range(n)
    ]

    args = {
        "target_col": "y",
        "feature_cols": ["x", "z", "segment"],
        "min_n_per_parameter": 3,
        "alpha": 0.05,
    }

    raw = plugin.run(LocalContext(df, args))

    assert raw["status"] in {"ok", "warning"}

    details = raw["details"]
    robust_summary = details.get("robust_se_summary", {}) or {}
    robust_available = robust_summary.get("available") is True

    # The core HC3 computation should now succeed.
    assert robust_available is True
    assert details.get("coef_table_robust_hc3")
    assert details.get("coef_table_classical_vs_robust")

    run = plugin.build_analysis_run(
        action_id="act_linear_model",
        arguments=args,
        data_version_id="data_v_test",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=details,
        artifacts=raw.get("artifacts", []),
        observation_id="obs_linear_model",
    )

    metric_value = run["metrics"].get("robust_se_available")
    summary = run.get("summary", "")

    assert metric_value is robust_available

    if "HC3 robust standard errors are reported" in summary:
        assert robust_available is True

    if not robust_available:
        assert "robust standard errors were not available" in summary