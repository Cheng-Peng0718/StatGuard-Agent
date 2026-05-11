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


def test_correlation_matrix_unified_execute_and_analysis_run():
    plugin = get_plugin("get_correlation_matrix")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [2, 4, 6, 8],
        "C": ["x", "y", "x", "z"],
    })

    raw = plugin.run(DummyContext(df, {"columns": ["A", "B", "C"]}))

    assert raw["status"] == "ok"
    assert raw["details"]["method"] == "pearson"
    assert raw["details"]["n_numeric_columns"] == 2
    assert raw["details"]["numeric_columns"] == ["A", "B"]
    assert "correlation_rows" in raw["details"]

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={"columns": ["A", "B", "C"]},
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=raw.get("artifacts", []),
        observation_id="obs_test",
    )

    assert run["tool_name"] == "get_correlation_matrix"
    assert run["title"] == "Correlation Matrix"
    assert run["metrics"]["method"] == "pearson"
    assert run["metrics"]["n_numeric_columns"] == 2
    assert "correlation_matrix" in run["tables"]
    assert run["report_blocks"]

    table_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "table"
    )

    labels = [col["label"] for col in table_block["columns"]]

    assert "Variable" in labels
    assert "A" in labels
    assert "B" in labels


def test_correlation_matrix_blocks_if_fewer_than_two_numeric_columns():
    plugin = get_plugin("get_correlation_matrix")

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "C": ["x", "y", "z"],
    })

    raw = plugin.run(DummyContext(df, {"columns": ["A", "C"]}))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "INSUFFICIENT_NUMERIC_COLUMNS"