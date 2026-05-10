from core.analysis_runs import build_analysis_run_from_observation
from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.display import (
    DisplayConfig,
    MetricDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin


def test_analysis_runs_prefers_unified_plugin():
    def extract_demo(
        *,
        payload,
        arguments,
        default_title,
        default_summary,
    ):
        title = "Unified Demo Analysis"
        summary = "Built from unified plugin."

        metrics = compact_dict({
            "value": payload.get("value"),
        })

        tables = {}
        metadata = {
            "source": "unified",
        }

        return title, summary, metrics, tables, metadata

    display_config = DisplayConfig(
        metrics=MetricDisplayConfig(
            labels={
                "value": "Demo value",
            },
            formatters={
                "value": lambda x: format_number(x, digits=2),
            },
            order=["value"],
        )
    )

    plugin = AnalysisToolPlugin(
        tool_name="unit_test_unified_analysis_run_tool",
        display_name="Unified Demo",
        extractor=extract_demo,
        display_config=display_config,
    )

    register_plugin(plugin)

    run = build_analysis_run_from_observation(
        tool_name="unit_test_unified_analysis_run_tool",
        action_id="act_test",
        arguments={},
        data_version_id="raw_v1",
        status="ok",
        success=True,
        message="Finished.",
        payload={"value": 1.23456},
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "unit_test_unified_analysis_run_tool"
    assert run["title"] == "Unified Demo Analysis"
    assert run["metadata"]["source"] == "unified"

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    assert metric_block["rows"][0]["label"] == "Demo value"
    assert metric_block["rows"][0]["value"] == "1.23"


def test_analysis_runs_unknown_tool_placeholder_still_works():
    run = build_analysis_run_from_observation(
        tool_name="get_summary_stats",
        action_id="act_test",
        arguments={},
        data_version_id="raw_v1",
        status="ok",
        success=True,
        message="Summary complete.",
        payload={
            "numeric_summary": {
                "A": {
                    "mean": 1.0,
                    "std": 2.0,
                }
            }
        },
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "get_summary_stats"
    assert run["report_blocks"]