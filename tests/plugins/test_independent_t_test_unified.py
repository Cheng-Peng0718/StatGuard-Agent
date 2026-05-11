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


def test_independent_t_test_unified_execute_and_analysis_run():
    plugin = get_plugin("run_independent_t_test")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "score": [1, 2, 3, 4, 8, 9, 10, 11],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "score",
            "group_col": "group",
            "group1_val": "A",
            "group2_val": "B",
        },
    ))

    assert raw["status"] == "ok"
    assert raw["details"]["method"] == "Welch two-sample t-test"
    assert raw["details"]["group1_n"] == 4
    assert raw["details"]["group2_n"] == 4
    assert "t_statistic" in raw["details"]
    assert "p_value" in raw["details"]

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "target_col": "score",
            "group_col": "group",
            "group1_val": "A",
            "group2_val": "B",
        },
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "run_independent_t_test"
    assert run["title"] == "Independent t-test: score by group"
    assert run["metrics"]["group1_n"] == 4
    assert run["metrics"]["group2_n"] == 4
    assert run["report_blocks"]

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Method" in labels
    assert "t statistic" in labels
    assert "p-value" in labels
    assert "Significant at 0.05" in labels


def test_independent_t_test_blocks_missing_column():
    plugin = get_plugin("run_independent_t_test")

    df = pd.DataFrame({
        "score": [1, 2, 3],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "score",
            "group_col": "group",
            "group1_val": "A",
            "group2_val": "B",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "COLUMNS_NOT_FOUND"


def test_independent_t_test_blocks_small_group():
    plugin = get_plugin("run_independent_t_test")

    df = pd.DataFrame({
        "score": [1, 2, 3],
        "group": ["A", "A", "B"],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "score",
            "group_col": "group",
            "group1_val": "A",
            "group2_val": "B",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "INSUFFICIENT_GROUP_SIZE"