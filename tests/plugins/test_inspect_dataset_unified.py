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


def test_inspect_dataset_unified_execute_and_analysis_run():
    plugin = get_plugin("inspect_dataset")

    assert plugin is not None
    assert plugin.execute is not None

    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": ["x", "unknown", "y"],
    })

    raw = plugin.run(DummyContext(df))

    assert raw["status"] == "ok"
    assert raw["details"]["shape"]["rows"] == 3
    assert raw["details"]["shape"]["columns"] == 2
    assert raw["details"]["total_missing"] >= 1
    assert len(raw["details"]["columns"]) == 2

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

    assert run["tool_name"] == "inspect_dataset"
    assert run["title"] == "Dataset Inspection"
    assert run["metrics"]["rows"] == 3
    assert run["metrics"]["columns"] == 2
    assert "columns" in run["tables"]
    assert run["report_blocks"]