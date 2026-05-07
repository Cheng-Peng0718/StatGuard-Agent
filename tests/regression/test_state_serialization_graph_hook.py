from pydantic import BaseModel

from core.graph import _attach_state_serialization_audit


class DummyModel(BaseModel):
    name: str
    value: int


def test_attach_state_serialization_audit_records_compact_warning():
    state = {
        "observations": [],
        "analysis_runs": [],
    }

    updates = {
        "current_action": DummyModel(
            name="clean_data",
            value=1,
        )
    }

    result = _attach_state_serialization_audit(state, updates)

    assert "state_serialization_audit" in result

    audit = result["state_serialization_audit"]

    assert audit["status"] == "warning"
    assert audit["n_issues"] >= 1
    assert any(
        issue["code"] == "PYDANTIC_MODEL_NORMALIZED"
        for issue in audit["issues"]
    )

    # Important: GraphState should not store a full duplicated safe_state.
    assert "safe_state" not in audit


def test_attach_state_serialization_audit_ok_for_plain_updates():
    state = {
        "observations": [],
        "analysis_runs": [],
    }

    updates = {
        "assistant_response": {
            "response_type": "final_answer",
            "content": "Done.",
        }
    }

    result = _attach_state_serialization_audit(state, updates)

    assert result["state_serialization_audit"]["status"] == "ok"
    assert result["state_serialization_audit"]["n_issues"] == 0
    assert result["state_serialization_audit"]["issues"] == []