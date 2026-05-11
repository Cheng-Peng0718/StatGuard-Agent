from core.controller.backend_turn import _apply_updates


def test_backend_turn_apply_updates_appends_observation_deltas():
    state = {
        "observations": [
            {
                "observation_id": "obs_old",
                "tool_name": "get_summary_stats",
            }
        ],
        "analysis_runs": [],
    }

    updates = {
        "observations": [
            {
                "observation_id": "obs_new",
                "tool_name": "missingness_report",
            }
        ]
    }

    result = _apply_updates(state, updates)

    assert [
        obs["observation_id"]
        for obs in result["observations"]
    ] == ["obs_old", "obs_new"]


def test_backend_turn_apply_updates_appends_analysis_runs_delta():
    old_run = {
        "analysis_run_id": "run_old",
        "observation_id": "obs_old",
        "tool_name": "get_summary_stats",
    }

    new_run = {
        "analysis_run_id": "run_new",
        "observation_id": "obs_new",
        "tool_name": "missingness_report",
    }

    state = {
        "analysis_runs": [old_run],
    }

    updates = {
        "analysis_runs": [new_run],
    }

    result = _apply_updates(state, updates)

    assert result["analysis_runs"] == [old_run, new_run]