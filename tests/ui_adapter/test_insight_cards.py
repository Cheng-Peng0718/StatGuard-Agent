import json

from core.ui_adapter.insight_cards import (
    build_insight_card_from_run,
    build_insight_cards_from_state,
    build_latest_insight_card_from_state,
)


def test_insight_card_for_successful_summary_run():
    run = {
        "analysis_run_id": "run_1",
        "observation_id": "obs_1",
        "tool_name": "get_summary_stats",
        "status": "ok",
        "success": True,
        "summary": "Computed summary statistics.",
        "metrics": {
            "n_rows": 5,
            "n_cols": 3,
        },
        "guardrails": [],
        "data_version_id": "raw_v1",
    }

    card = build_insight_card_from_run(run)

    assert card["title"] == "Summary Statistics"
    assert card["success"] is True
    assert "Computed summary statistics." in card["key_findings"]
    assert any("Rows: 5" in item for item in card["key_findings"])
    assert any("Columns: 3" in item for item in card["key_findings"])
    assert card["recommended_next_steps"]

    json.dumps(card)


def test_insight_card_for_failed_regression_run():
    run = {
        "analysis_run_id": "run_reg",
        "observation_id": "obs_reg",
        "tool_name": "run_multiple_regression",
        "status": "failed",
        "success": False,
        "error_code": "INSUFFICIENT_SAMPLE_SIZE",
        "message": "Regression failed due to insufficient sample size.",
        "data_version_id": "raw_v1",
    }

    card = build_insight_card_from_run(run)

    assert card["title"] == "Multiple Regression"
    assert card["success"] is False
    assert any(
        "INSUFFICIENT_SAMPLE_SIZE" in item
        for item in card["key_findings"]
    )
    assert any("should not be interpreted" in item for item in card["caveats"])
    assert any("sample size" in item for item in card["recommended_next_steps"])

    json.dumps(card)


def test_insight_cards_from_state_and_latest_card():
    state = {
        "analysis_runs": [
            {
                "analysis_run_id": "run_1",
                "observation_id": "obs_1",
                "tool_name": "missingness_report",
                "status": "ok",
                "success": True,
                "summary": "Missingness report completed.",
            },
            {
                "analysis_run_id": "run_2",
                "observation_id": "obs_2",
                "tool_name": "get_correlation_matrix",
                "status": "ok",
                "success": True,
                "summary": "Correlation matrix completed.",
            },
        ]
    }

    cards = build_insight_cards_from_state(state)
    latest = build_latest_insight_card_from_state(state)

    assert len(cards) == 2
    assert latest["tool_name"] == "get_correlation_matrix"

    json.dumps(cards)
    json.dumps(latest)