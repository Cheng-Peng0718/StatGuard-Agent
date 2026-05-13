from pathlib import Path

import pandas as pd

from core.analysis_tool_plugins import get_plugin
from core.analysis_tool_plugins.registry import get_tool_specs_for_llm


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
            "customer_id": [1, 2, 3, 4],
            "region": ["East", "East", "West", "West"],
            "number_of_orders": [1, 2, 3, 4],
            "total_revenue": [100.0, 200.0, 300.0, 400.0],
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


def test_core_plugins_declare_evidence_categories():
    expected = {
        "inspect_sql_schema": {"sql_schema"},
        "materialize_sql_query_result": {"data_preparation"},
        "kpi_summary": {"kpi_summary"},
        "statistical_group_comparison": {"group_comparison", "statistical_inference"},
        "run_multiple_regression": {"regression_model", "statistical_inference"},
        "regression_diagnostics": {"regression_diagnostics"},
    }

    for tool_name, required_categories in expected.items():
        plugin = get_plugin(tool_name)
        assert plugin is not None, f"Missing plugin: {tool_name}"

        categories = set(plugin.evidence_categories or [])
        assert required_categories.issubset(categories), (
            f"{tool_name} evidence_categories={categories}, "
            f"expected to include {required_categories}"
        )


def test_tool_specs_expose_evidence_categories_to_supervisor():
    specs = get_tool_specs_for_llm()

    assert "run_multiple_regression" in specs
    assert "evidence_categories" in specs["run_multiple_regression"]
    assert "regression_model" in specs["run_multiple_regression"]["evidence_categories"]

    assert "kpi_summary" in specs
    assert "kpi_summary" in specs["kpi_summary"]["evidence_categories"]


def test_analysis_run_records_plugin_evidence_categories(tmp_path):
    plugin = get_plugin("kpi_summary")
    assert plugin is not None

    data_versions, active_id = _make_active_dataset(tmp_path)

    raw = plugin.run(
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

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "metric_columns": ["total_revenue", "number_of_orders"],
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
    assert "evidence_categories" in run
    assert "kpi_summary" in run["evidence_categories"]