from pathlib import Path

import pandas as pd

from core.analysis_tool_plugins import get_plugin


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


def test_kpi_summary_with_explicit_metrics(tmp_path):
    plugin = get_plugin("kpi_summary")
    assert plugin is not None

    data_versions, active_id = _make_active_dataset(tmp_path)

    result = plugin.run(
        DummyContext(
            {
                "metric_columns": ["total_revenue", "number_of_orders"],
                "id_columns": ["customer_id"],
            },
            tmp_path,
            data_versions,
            active_id,
        )
    )

    assert result["status"] == "ok"

    details = result["details"]
    assert details["n_rows"] == 6
    assert details["metric_columns"] == ["total_revenue", "number_of_orders"]

    revenue_row = next(
        row for row in details["kpi_rows"]
        if row["metric"] == "total_revenue"
    )

    assert revenue_row["total"] == 1520.0
    assert round(revenue_row["mean"], 6) == round(1520.0 / 6, 6)

    distinct_row = details["distinct_count_rows"][0]
    assert distinct_row["column"] == "customer_id"
    assert distinct_row["distinct_count"] == 6


def test_kpi_summary_infers_numeric_metrics(tmp_path):
    plugin = get_plugin("kpi_summary")
    assert plugin is not None

    data_versions, active_id = _make_active_dataset(tmp_path)

    result = plugin.run(
        DummyContext({}, tmp_path, data_versions, active_id)
    )

    assert result["status"] == "ok"
    assert "total_revenue" in result["details"]["metric_columns"]
    assert "customer_id" not in result["details"]["metric_columns"]
    assert "customer_id" in result["details"]["id_columns"]


def test_kpi_summary_blocks_missing_column(tmp_path):
    plugin = get_plugin("kpi_summary")
    assert plugin is not None

    data_versions, active_id = _make_active_dataset(tmp_path)

    result = plugin.run(
        DummyContext(
            {"metric_columns": ["bad_metric"]},
            tmp_path,
            data_versions,
            active_id,
        )
    )

    assert result["status"] == "blocked"
    assert result["error_code"] == "COLUMN_NOT_FOUND"


def test_kpi_summary_builds_analysis_run(tmp_path):
    plugin = get_plugin("kpi_summary")
    assert plugin is not None

    data_versions, active_id = _make_active_dataset(tmp_path)

    raw = plugin.run(
        DummyContext(
            {
                "metric_columns": ["total_revenue"],
                "id_columns": ["customer_id"],
            },
            tmp_path,
            data_versions,
            active_id,
        )
    )

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "metric_columns": ["total_revenue"],
            "id_columns": ["customer_id"],
        },
        data_version_id=active_id,
        status=raw["status"],
        success=True,
        message=raw["message"],
        payload=raw["details"],
        artifacts=raw.get("artifacts", []),
        observation_id="obs_test",
    )

    assert run["tool_name"] == "kpi_summary"
    assert run["title"] == "KPI Summary"
    assert run["metrics"]["n_rows"] == 6
    assert "kpi_rows" in run["tables"]
    assert run["report_blocks"]