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


def test_residual_histogram_unified_execute_and_analysis_run(tmp_path):
    plugin = get_plugin("generate_residual_histogram")

    assert plugin is not None
    assert plugin.execute is not None

    output_path = tmp_path / "residual_histogram.png"

    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "x": [2, 4, 6, 8, 10, 12, 14, 16],
    })

    raw = plugin.run(DummyContext(
        df,
        {
            "target_col": "y",
            "feature_cols": ["x"],
            "output_path": str(output_path),
            "min_n_per_parameter": 1,
        },
        workspace_dir=str(tmp_path),
    ))

    assert raw["status"] in {"ok", "warning"}
    assert output_path.exists()

    assert "n_residuals" in raw["details"]
    assert "residual_summary" in raw["details"]
    assert "diagnostic_flags" in raw["details"]
    assert raw["artifacts"]
    assert raw["artifacts"][0]["type"] == "png"

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "target_col": "y",
            "feature_cols": ["x"],
            "output_path": str(output_path),
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

    assert run["tool_name"] == "generate_residual_histogram"
    assert run["title"] == "Residual Histogram"

    assert "n_residuals" in run["metrics"]
    assert "residual_mean" in run["metrics"]
    assert "residual_summary" in run["metadata"]

    assert run["artifacts"]
    assert run["artifacts"][0]["type"] == "png"

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]

    assert "Residual count" in labels
    assert "Residual mean" in labels
    assert "Residual SD" in labels

    figure_blocks = [
        block for block in run["report_blocks"]
        if block["type"] == "figure"
    ]

    assert figure_blocks
    assert Path(figure_blocks[0]["path"]).exists()


def test_residual_histogram_blocks_missing_feature_column():
    plugin = get_plugin("generate_residual_histogram")

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