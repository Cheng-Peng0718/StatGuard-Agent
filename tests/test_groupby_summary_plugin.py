from pathlib import Path

import pandas as pd


class DummyContext:
    def __init__(self, arguments, workspace_dir, data_versions, active_data_version_id):
        self.arguments = arguments
        self.args = arguments
        self.workspace_dir = str(workspace_dir)
        self.data_versions = data_versions
        self.active_data_version_id = active_data_version_id
        self.data_audit_log = []


def _make_active_dataset(tmp_path: Path):
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "region": ["East", "East", "West", "West", "West", "South"],
            "segment": ["Consumer", "Corporate", "Consumer", "Consumer", "Corporate", "Consumer"],
            "number_of_orders": [3, 2, 5, 1, 4, 2],
            "total_revenue": [300.0, 150.0, 500.0, 50.0, 400.0, 120.0],
        }
    )

    path = tmp_path / "active.parquet"
    df.to_parquet(path, index=False)

    version = {
        "version_id": "data_v_test",
        "parent_version_id": None,
        "path": str(path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "operation": "test_dataset",
    }

    return [version], "data_v_test"


def test_groupby_summary_by_region(tmp_path):
    import core.analysis_tool_plugins.plugins  # noqa: F401
    from core.analysis_tool_plugins.registry import get_plugin

    data_versions, active_id = _make_active_dataset(tmp_path)

    plugin = get_plugin("groupby_summary")
    assert plugin is not None

    result = plugin.run(
        DummyContext(
            {
                "group_cols": ["region"],
                "value_col": "total_revenue",
                "agg_funcs": ["count", "sum", "mean"],
                "sort_by": "sum_total_revenue",
                "ascending": False,
            },
            tmp_path,
            data_versions,
            active_id,
        )
    )

    assert result["status"] == "ok"

    details = result["details"]
    assert details["value_col"] == "total_revenue"
    assert details["group_cols"] == ["region"]
    assert details["n_groups"] == 3

    rows = details["rows"]
    assert rows[0]["region"] == "West"
    assert rows[0]["sum_total_revenue"] == 950.0


def test_groupby_summary_blocks_missing_column(tmp_path):
    import core.analysis_tool_plugins.plugins  # noqa: F401
    from core.analysis_tool_plugins.registry import get_plugin

    data_versions, active_id = _make_active_dataset(tmp_path)

    plugin = get_plugin("groupby_summary")
    assert plugin is not None

    result = plugin.run(
        DummyContext(
            {
                "group_cols": ["bad_region"],
                "value_col": "total_revenue",
            },
            tmp_path,
            data_versions,
            active_id,
        )
    )

    assert result["status"] == "blocked"
    assert result["error_code"] == "COLUMN_NOT_FOUND"


def test_groupby_summary_blocks_without_active_dataset(tmp_path):
    import core.analysis_tool_plugins.plugins  # noqa: F401
    from core.analysis_tool_plugins.registry import get_plugin

    plugin = get_plugin("groupby_summary")
    assert plugin is not None

    result = plugin.run(
        DummyContext(
            {
                "group_cols": ["region"],
                "value_col": "total_revenue",
            },
            tmp_path,
            [],
            None,
        )
    )

    assert result["status"] == "failed"
    assert result["error_code"] == "GROUPBY_SUMMARY_FAILED"