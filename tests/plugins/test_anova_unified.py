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


def test_anova_unified_execute_and_analysis_run():
    plugin = get_plugin("run_anova")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "score": [1, 2, 3, 4, 8, 9, 10, 11, 4, 5, 6, 7],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "score",
            "group_col": "group",
        },
    ))

    assert raw["status"] == "ok"
    assert raw["details"]["method"] == "One-way ANOVA"
    assert raw["details"]["valid_group_count"] == 3
    assert raw["details"]["nobs"] == 12
    assert "F_statistic" in raw["details"]
    assert "p_value" in raw["details"]
    assert "group_summaries" in raw["details"]

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "target_col": "score",
            "group_col": "group",
        },
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "run_anova"
    assert run["title"] == "One-way ANOVA: score by group"
    assert run["metrics"]["valid_group_count"] == 3
    assert "group_summaries" in run["tables"]
    assert run["report_blocks"]

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Method" in labels
    assert "F statistic" in labels
    assert "p-value" in labels
    assert "Significant at 0.05" in labels

    table_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "table"
    )

    table_labels = [col["label"] for col in table_block["columns"]]

    assert "Group" in table_labels
    assert "Mean" in table_labels
    assert "SD" in table_labels


def test_anova_blocks_missing_column():
    plugin = get_plugin("run_anova")

    df = pd.DataFrame({
        "score": [1, 2, 3],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "score",
            "group_col": "group",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "COLUMNS_NOT_FOUND"


def test_anova_blocks_insufficient_groups():
    plugin = get_plugin("run_anova")

    df = pd.DataFrame({
        "score": [1, 2, 3, 4],
        "group": ["A", "A", "A", "A"],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "score",
            "group_col": "group",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "INSUFFICIENT_GROUPS"