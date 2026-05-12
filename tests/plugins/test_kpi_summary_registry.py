from core.analysis_tool_plugins import get_plugin
from core.analysis_tool_plugins.registry import get_tool_specs_for_llm


def test_kpi_summary_is_registered():
    plugin = get_plugin("kpi_summary")

    assert plugin is not None
    assert plugin.tool_name == "kpi_summary"
    assert plugin.display_name == "KPI Summary"
    assert plugin.requires_data_source == "dataframe"
    assert plugin.execute is not None


def test_kpi_summary_is_visible_to_supervisor_tool_cards():
    specs = get_tool_specs_for_llm()

    assert "kpi_summary" in specs

    spec = specs["kpi_summary"]

    assert spec["name"] == "kpi_summary"
    assert spec["display_name"] == "KPI Summary"
    assert spec["requires_data_source"] == "dataframe"
    assert spec["produces_active_dataset"] is False
    assert "metric_columns" in spec["argument_schema"]["optional"]
    assert "id_columns" in spec["argument_schema"]["optional"]