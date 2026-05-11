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


def test_summarize_columns_unified_execute_and_analysis_run():
    plugin = get_plugin("summarize_columns")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "x", "y"],
    })

    raw = plugin.run(DummyContext(df, {"columns": ["A", "B"]}))

    assert raw["status"] == "ok"
    assert raw["details"]["n_columns_summarized"] == 2
    assert raw["details"]["resolved_columns"] == ["A", "B"]
    assert len(raw["details"]["summary_rows"]) == 2

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={"columns": ["A", "B"]},
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=raw.get("artifacts", []),
        observation_id="obs_test",
    )

    assert run["tool_name"] == "summarize_columns"
    assert run["title"] == "Column Summary: A, B"
    assert run["metrics"]["n_columns_summarized"] == 2
    assert "summary_rows" in run["tables"]
    assert run["report_blocks"]

    table_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "table"
    )

    labels = [col["label"] for col in table_block["columns"]]

    assert "Column" in labels
    assert "Mean" in labels
    assert "Top values" in labels


def test_summarize_columns_blocks_missing_column():
    plugin = get_plugin("summarize_columns")

    df = pd.DataFrame({
        "A": [1, 2, 3],
    })

    raw = plugin.run(DummyContext(df, {"columns": ["Z"]}))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "COLUMNS_NOT_FOUND"