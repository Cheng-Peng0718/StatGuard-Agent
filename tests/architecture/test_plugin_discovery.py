from core.analysis_tool_plugins import PLUGIN_REGISTRY, get_plugin


def test_core_unified_plugins_are_auto_discovered():
    expected_plugins = {
        "clean_data",
        "generate_scatterplot",
        "generate_residual_histogram",
        "get_correlation_matrix",
        "get_summary_stats",
        "inspect_dataset",
        "missingness_report",
        "regression_diagnostics",
        "run_anova",
        "run_chi_square",
        "run_correlation_test",
        "run_independent_t_test",
        "run_multiple_regression",
        "summarize_columns",
    }

    missing = expected_plugins - set(PLUGIN_REGISTRY.keys())

    assert not missing, (
        f"Expected unified plugins to be auto-discovered, but missing: {missing}"
    )


def test_unknown_tool_falls_back_to_generic_unified_plugin():
    from core.analysis_runs import build_analysis_run_from_observation

    run = build_analysis_run_from_observation(
        tool_name="unknown_tool_demo",
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

    assert run["tool_name"] == "unknown_tool_demo"
    assert run["title"] == "Unknown Tool Demo"
    assert run["status"] == "ok"
    assert run["report_blocks"]