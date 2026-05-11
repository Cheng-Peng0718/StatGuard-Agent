from core.analysis_runs import build_analysis_run_from_observation


def test_analysis_run_preserves_successful_observation_core_fields():
    observation = {
        "observation_id": "obs_123",
        "tool_name": "get_summary_stats",
        "arguments": {},
        "status": "ok",
        "success": True,
        "error_code": None,
        "message": "Summary statistics computed.",
        "summary": "Summary statistics computed.",
        "data_version_id": "raw_v1",
        "artifacts": [],
        "structured_data": {
            "rows": 226,
            "columns": 25,
        },
        "raw_data": {
            "payload": "raw",
        },
    }

    run = build_analysis_run_from_observation(observation=observation)

    assert run["observation_id"] == "obs_123"
    assert run["tool_name"] == "get_summary_stats"
    assert run["arguments"] == {}
    assert run["status"] == "ok"
    assert run["success"] is True
    assert run["error_code"] is None
    assert run["data_version_id"] == "raw_v1"
    assert run["artifacts"] == []
    assert run["summary"] == "Summary statistics computed."


def test_analysis_run_preserves_artifacts():
    observation = {
        "observation_id": "obs_plot",
        "tool_name": "generate_scatterplot",
        "arguments": {
            "x_column": "SATM",
            "y_column": "GPA",
        },
        "status": "ok",
        "success": True,
        "data_version_id": "raw_v1",
        "artifacts": [
            {
                "artifact_type": "plot",
                "path": "plots/scatter.png",
            }
        ],
        "structured_data": {},
        "raw_data": {},
    }

    run = build_analysis_run_from_observation(observation=observation)

    assert run["tool_name"] == "generate_scatterplot"
    assert run["artifacts"] == [
        {
            "artifact_type": "plot",
            "path": "plots/scatter.png",
        }
    ]


def test_analysis_run_preserves_failed_observation_status():
    observation = {
        "observation_id": "obs_failed",
        "tool_name": "run_multiple_regression",
        "arguments": {
            "target_col": "GPA",
            "feature_cols": ["SATM"],
        },
        "status": "failed",
        "success": False,
        "error_code": "SINGULAR_MATRIX",
        "message": "Model failed.",
        "summary": "Model failed.",
        "data_version_id": "raw_v1",
        "artifacts": [],
        "structured_data": {},
        "raw_data": {},
    }

    run = build_analysis_run_from_observation(observation=observation)

    assert run["observation_id"] == "obs_failed"
    assert run["tool_name"] == "run_multiple_regression"
    assert run["status"] == "failed"
    assert run["success"] is False
    assert run["error_code"] == "SINGULAR_MATRIX"
    assert run["message"] == "Model failed."


def test_analysis_run_defaults_missing_arguments_to_empty_dict():
    observation = {
        "observation_id": "obs_no_args",
        "tool_name": "get_summary_stats",
        "status": "ok",
        "success": True,
        "data_version_id": "raw_v1",
        "artifacts": [],
        "structured_data": {},
        "raw_data": {},
    }

    run = build_analysis_run_from_observation(observation=observation)

    assert run["arguments"] == {}


def test_analysis_run_keeps_guardrails_if_present():
    observation = {
        "observation_id": "obs_reg",
        "tool_name": "run_multiple_regression",
        "arguments": {
            "target_col": "GPA",
            "feature_cols": ["SATM"],
        },
        "status": "ok",
        "success": True,
        "data_version_id": "raw_v1",
        "artifacts": [],
        "structured_data": {},
        "raw_data": {},
        "guardrails": [
            {
                "severity": "warning",
                "code": "NON_NORMAL_RESIDUALS",
                "message": "Residuals may be non-normal.",
            }
        ],
    }

    run = build_analysis_run_from_observation(observation=observation)

    assert run["guardrails"] == [
        {
            "severity": "warning",
            "code": "NON_NORMAL_RESIDUALS",
            "message": "Residuals may be non-normal.",
        }
    ]

def test_analysis_run_observation_contract_keeps_plugin_summary_separate():
    observation = {
        "observation_id": "obs_123",
        "tool_name": "get_summary_stats",
        "arguments": {},
        "status": "ok",
        "success": True,
        "message": "Canonical message.",
        "summary": "Canonical summary.",
        "data_version_id": "raw_v1",
        "artifacts": [],
        "structured_data": {},
        "raw_data": {},
    }

    run = build_analysis_run_from_observation(observation=observation)

    assert run["summary"] == "Canonical summary."
    assert "plugin_summary" in run

def test_analysis_run_failed_observation_still_builds_with_sparse_payload():
    observation = {
        "observation_id": "obs_failed_sparse",
        "tool_name": "run_multiple_regression",
        "arguments": {
            "target_col": "GPA",
            "feature_cols": ["SATM"],
        },
        "status": "failed",
        "success": False,
        "error_code": "MODEL_FIT_FAILED",
        "message": "Model fitting failed before producing a payload.",
        "summary": "Model fitting failed.",
        "data_version_id": "raw_v1",
        "artifacts": [],
        "structured_data": {},
        "raw_data": {},
    }

    run = build_analysis_run_from_observation(observation=observation)

    assert run["observation_id"] == "obs_failed_sparse"
    assert run["tool_name"] == "run_multiple_regression"
    assert run["status"] == "failed"
    assert run["success"] is False
    assert run["error_code"] == "MODEL_FIT_FAILED"
    assert run["data_version_id"] == "raw_v1"