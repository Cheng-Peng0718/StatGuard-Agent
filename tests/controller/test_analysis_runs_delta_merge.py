from core.controller.backend_turn import _apply_updates


def test_backend_controller_appends_analysis_runs_delta():
    state = {
        "analysis_runs": [
            {
                "analysis_run_id": "run_1",
                "tool_name": "get_summary_stats",
            }
        ]
    }

    updates = {
        "analysis_runs": [
            {
                "analysis_run_id": "run_2",
                "tool_name": "run_multiple_regression",
            }
        ]
    }

    merged = _apply_updates(state, updates)

    assert [run["analysis_run_id"] for run in merged["analysis_runs"]] == [
        "run_1",
        "run_2",
    ]


def test_backend_controller_appends_observations_delta_still_works():
    state = {
        "observations": [
            {
                "observation_id": "obs_1",
            }
        ]
    }

    updates = {
        "observations": [
            {
                "observation_id": "obs_2",
            }
        ]
    }

    merged = _apply_updates(state, updates)

    assert [obs["observation_id"] for obs in merged["observations"]] == [
        "obs_1",
        "obs_2",
    ]