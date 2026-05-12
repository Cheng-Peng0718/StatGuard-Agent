from core.analysis_tool_plugins.base import AnalysisToolPlugin


def test_blocked_plugin_result_builds_generic_report_blocks():
    plugin = AnalysisToolPlugin(
        tool_name="example_tool",
        display_name="Example Tool",
    )

    run = plugin.build_analysis_run(
        action_id="act_blocked",
        arguments={"x": "bad_column"},
        data_version_id="data_v_test",
        status="blocked",
        success=False,
        message="The requested analysis could not be completed.",
        payload={
            "error_code": "INSUFFICIENT_GROUPS",
            "reason": "Each group needs at least two observations.",
            "group_summaries": [
                {"group": "East", "n": 1},
                {"group": "West", "n": 1},
            ],
        },
        artifacts=[],
        observation_id="obs_blocked",
    )

    assert run["status"] == "blocked"
    assert run["success"] is False
    assert run["title"] == "Example Tool (Blocked)"
    assert run["metrics"]["status"] == "blocked"
    assert "details" in run["tables"]
    assert run["report_blocks"]