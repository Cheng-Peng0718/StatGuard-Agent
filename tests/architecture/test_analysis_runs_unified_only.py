from core.analysis_runs import build_analysis_run_from_observation


def test_analysis_runs_unknown_tool_uses_generic_placeholder_plugin():
    run = build_analysis_run_from_observation(
        tool_name="unknown_tool_after_migration",
        action_id="act_test",
        arguments={},
        data_version_id="raw_v1",
        status="ok",
        success=True,
        message="Fallback test.",
        payload={"hello": "world"},
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "unknown_tool_after_migration"
    assert run["title"] == "Unknown Tool After Migration"
    assert run["report_blocks"]

    block_types = [block["type"] for block in run["report_blocks"]]
    assert "json" in block_types or "text" in block_types