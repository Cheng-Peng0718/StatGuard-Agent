from core.analysis_tool_plugins import get_plugin


def test_correlation_test_plugin_builds_analysis_run():
    plugin = get_plugin("run_correlation_test")

    assert plugin is not None

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={
            "x_col": "GPA",
            "y_col": "SATM",
            "method": "pearson",
        },
        data_version_id="raw_v1",
        status="ok",
        success=True,
        message="Correlation test completed.",
        payload={
            "x_col": "GPA",
            "y_col": "SATM",
            "method": "pearson",
            "nobs": 216,
            "correlation": 0.2672,
            "p_value": 0.00007,
            "significant_at_0_05": True,
        },
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "run_correlation_test"
    assert run["title"] == "Correlation Test: GPA vs SATM"

    assert run["metrics"]["method"] == "pearson"
    assert run["metrics"]["nobs"] == 216
    assert run["metrics"]["correlation"] == 0.2672
    assert run["metrics"]["p_value"] == 0.00007

    assert run["metadata"]["x_col"] == "GPA"
    assert run["metadata"]["y_col"] == "SATM"
    assert run["metadata"]["significant_at_0_05"] is True

    metric_block = next(
        block for block in run["report_blocks"]
        if block["type"] == "metric_table"
    )

    labels = [row["label"] for row in metric_block["rows"]]
    values = [row["value"] for row in metric_block["rows"]]

    assert "Method" in labels
    assert "Observations used" in labels
    assert "Correlation coefficient" in labels
    assert "p-value" in labels

    assert "<0.0001" in values