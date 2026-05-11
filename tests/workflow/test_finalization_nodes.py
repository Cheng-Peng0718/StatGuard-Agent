from core.workflow.nodes.finalization import (
    deliverable_gate_node,
    final_response_node,
)


def test_final_response_node_wraps_final_answer():
    result = final_response_node({
        "active_data_version_id": "raw_v1",
        "final_answer": "Final result.",
        "deliverable_check": {
            "status": "ok",
        },
    })

    assert result["assistant_response"]["response_type"] == "final_answer"
    assert result["assistant_response"]["content"] == "Final result."
    assert result["assistant_response"]["source_node"] == "final_response"

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None


def test_final_response_node_returns_error_when_content_missing():
    result = final_response_node({
        "active_data_version_id": "raw_v1",
        "deliverable_check": {
            "status": "ok",
        },
    })

    assert result["assistant_response"]["response_type"] == "error"
    assert result["assistant_response"]["metadata"]["reason"] == (
        "missing_final_answer_content"
    )

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None


def test_deliverable_gate_node_stores_deliverable_check():
    result = deliverable_gate_node({
        "analysis_runs": [],
        "task_contract": None,
    })

    assert "deliverable_check" in result
    assert result["deliverable_check"]["status"] in {
        "ok",
        "needs_more_work",
        "missing",
        "blocked",
    }