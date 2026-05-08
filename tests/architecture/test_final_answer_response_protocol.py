from pathlib import Path

from core.graph import final_response_node
from core.workflow.routes import route_after_deliverable_gate


def test_final_response_node_wraps_final_answer_in_assistant_response():
    state = {
        "active_data_version_id": "raw_v1",
        "final_answer": "This is the final result.",
        "deliverable_check": {
            "status": "ok",
            "message": "Passed.",
        },
    }

    result = final_response_node(state)

    assert "assistant_response" in result
    assert result["assistant_response"]["response_type"] == "final_answer"
    assert result["assistant_response"]["content"] == "This is the final result."
    assert result["assistant_response"]["source_node"] == "final_response"

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None


def test_deliverable_gate_routes_to_final_response():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    assert 'workflow.add_node("final_response"' in graph_text
    assert '"final_response": "final_response"' in graph_text
    assert 'workflow.add_edge("final_response", END)' in graph_text


def test_graph_has_final_response_node_and_no_direct_final_answer_edge():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")
    routes_text = Path("core/workflow/routes.py").read_text(encoding="utf-8")

    assert 'workflow.add_node("final_response"' in graph_text
    assert 'workflow.add_edge("final_response", END)' in graph_text

    # Deliverable gate success should route to final_response, not directly to END.
    assert "def route_after_deliverable_gate" in routes_text
    assert route_after_deliverable_gate({
        "deliverable_check": {
            "status": "ok",
        }
    }) == "final_response"