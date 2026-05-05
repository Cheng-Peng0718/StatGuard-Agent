from pathlib import Path

import pandas as pd

from core.analysis_tool_plugins import get_plugin


class DummyContext:
    def __init__(self, df, args=None, workspace_dir="."):
        self.df = df
        self.args = args or {}
        self.workspace_dir = workspace_dir

    def load_df(self):
        return self.df

    def get_arg(self, key, default=None):
        return self.args.get(key, default)


def test_scatterplot_unified_execute_and_analysis_run(tmp_path):
    plugin = get_plugin("generate_scatterplot")

    assert plugin is not None
    assert plugin.execute is not None

    output_path = tmp_path / "scatterplot.png"

    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "x_column": "x",
            "y_column": "y",
            "output_path": str(output_path),
        },
        workspace_dir=str(tmp_path),
    ))

    assert raw["status"] == "ok"
    assert output_path.exists()

    assert raw["details"]["x_column"] == "x"
    assert raw["details"]["y_column"] == "y"
    assert raw["details"]["nobs"] == 4
    assert raw["artifacts"]
    assert raw["artifacts"][0]["type"] == "png"

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "x_column": "x",
            "y_column": "y",
            "output_path": str(output_path),
        },
        data_version_id="raw_v1",
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=raw.get("artifacts", []),
        observation_id="obs_test",
    )

    assert run["tool_name"] == "generate_scatterplot"
    assert run["title"] == "Scatterplot: y vs x"

    assert run["metrics"]["nobs"] == 4
    assert run["artifacts"]
    assert run["artifacts"][0]["type"] == "png"
    assert run["report_blocks"]

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Observations plotted" in labels
    assert "X minimum" in labels
    assert "Y maximum" in labels

    figure_blocks = [
        block for block in run["report_blocks"]
        if block["type"] == "figure"
    ]

    assert figure_blocks
    assert Path(figure_blocks[0]["path"]).exists()


def test_scatterplot_blocks_missing_column():
    plugin = get_plugin("generate_scatterplot")

    df = pd.DataFrame({
        "x": [1, 2, 3],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "x_column": "x",
            "y_column": "y",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "COLUMNS_NOT_FOUND"


def test_scatterplot_blocks_insufficient_observations():
    plugin = get_plugin("generate_scatterplot")

    df = pd.DataFrame({
        "x": [1, None],
        "y": [2, None],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "x_column": "x",
            "y_column": "y",
        },
    ))

    assert raw["status"] == "blocked"
    assert raw["error_code"] == "INSUFFICIENT_VALID_OBSERVATIONS"