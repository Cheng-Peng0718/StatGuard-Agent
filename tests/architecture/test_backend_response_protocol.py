from pathlib import Path

from core.graph import final_response_node


def test_backend_final_response_uses_assistant_response():
    state = {
        "active_data_version_id": "raw_v1",
        "final_answer": "Final backend answer.",
        "deliverable_check": {
            "status": "ok",
            "message": "Passed.",
        },
    }

    result = final_response_node(state)

    assert "assistant_response" in result
    assert result["assistant_response"]["response_type"] == "final_answer"
    assert result["assistant_response"]["content"] == "Final backend answer."


def test_new_backend_response_nodes_do_not_return_final_answer_directly():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    for fn_name in [
        "advisory_answer_node",
        "plan_only_node",
        "execute_pending_plan_node",
    ]:
        start = graph_text.index(f"def {fn_name}")
        rest = graph_text[start + 1:]
        next_def_offset = rest.find("\ndef ")
        body = graph_text[start:] if next_def_offset == -1 else graph_text[start:start + 1 + next_def_offset]

        assert '"final_answer"' not in body
        assert "'final_answer'" not in body