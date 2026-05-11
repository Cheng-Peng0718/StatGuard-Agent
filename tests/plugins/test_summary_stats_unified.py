import pandas as pd

from core.analysis_tool_plugins import get_plugin


class DummyContext:
    def __init__(self, df):
        self.df = df
        self.args = {}

    def load_df(self):
        return self.df

    def get_arg(self, key, default=None):
        return self.args.get(key, default)


def test_summary_stats_unified_execute_and_analysis_run():
    plugin = get_plugin("get_summary_stats")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "x", "y"],
    })

    raw = plugin.run(DummyContext(df))

    assert raw["status"] == "ok"
    assert raw["details"]["n_rows"] == 3
    assert raw["details"]["n_columns"] == 2
    assert "numeric_summary" in raw["details"]
    assert "categorical_summary" in raw["details"]

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={},
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=raw.get("artifacts", []),
        observation_id="obs_test",
    )

    assert run["tool_name"] == "get_summary_stats"
    assert run["title"] == "Summary Statistics"
    assert run["metrics"]["n_rows"] == 3
    assert run["metrics"]["n_columns"] == 2
    assert run["report_blocks"]