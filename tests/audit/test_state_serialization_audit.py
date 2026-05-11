import json
from pathlib import Path

from pydantic import BaseModel

from core.audit.state_serialization import (
    audit_state_serialization,
    make_checkpoint_safe_state,
    to_jsonable,
)


class DummyModel(BaseModel):
    name: str
    value: int


class UnsupportedObject:
    def __init__(self):
        self.x = 1


def test_state_serialization_audit_ok_for_plain_dict():
    state = {
        "assistant_response": {
            "response_type": "final_answer",
            "content": "Done.",
        },
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "status": "ok",
            }
        ],
    }

    result = audit_state_serialization(state)

    assert result.status == "ok"
    assert result.issues == []

    json.dumps(result.safe_state)


def test_state_serialization_audit_normalizes_pydantic_model():
    state = {
        "current_action": DummyModel(
            name="clean_data",
            value=1,
        )
    }

    result = audit_state_serialization(state)

    assert result.status == "warning"
    assert result.safe_state["current_action"] == {
        "name": "clean_data",
        "value": 1,
    }

    assert any(
        issue.code == "PYDANTIC_MODEL_NORMALIZED"
        for issue in result.issues
    )

    json.dumps(result.safe_state)


def test_state_serialization_audit_normalizes_nested_pydantic_model():
    state = {
        "pending_plan": {
            "steps": [
                {
                    "step_id": "s1",
                    "action": DummyModel(name="tool", value=2),
                }
            ]
        }
    }

    result = audit_state_serialization(state)

    assert result.status == "warning"
    assert result.safe_state["pending_plan"]["steps"][0]["action"] == {
        "name": "tool",
        "value": 2,
    }

    json.dumps(result.safe_state)


def test_state_serialization_audit_normalizes_path_tuple_and_set():
    state = {
        "workspace_dir": Path("workspaces/test"),
        "tuple_value": ("a", "b"),
        "set_value": {"x", "y"},
    }

    result = audit_state_serialization(state)

    assert result.status == "warning"

    assert result.safe_state["workspace_dir"] == str(Path("workspaces/test"))
    assert result.safe_state["tuple_value"] == ["a", "b"]
    assert sorted(result.safe_state["set_value"]) == ["x", "y"]

    assert any(issue.code == "PATH_NORMALIZED" for issue in result.issues)
    assert any(issue.code == "TUPLE_NORMALIZED" for issue in result.issues)
    assert any(issue.code == "SET_NORMALIZED" for issue in result.issues)

    json.dumps(result.safe_state)


def test_state_serialization_audit_flags_unsupported_custom_object():
    state = {
        "bad_object": UnsupportedObject(),
    }

    result = audit_state_serialization(state)

    assert result.status == "error"
    assert any(
        issue.code == "UNSUPPORTED_CUSTOM_OBJECT"
        for issue in result.issues
    )

    json.dumps(result.safe_state)


def test_make_checkpoint_safe_state_returns_json_safe_dict():
    state = {
        "current_action": DummyModel(name="tool", value=3),
        "workspace_dir": Path("workspaces/test"),
    }

    safe_state = make_checkpoint_safe_state(state)

    assert safe_state["current_action"] == {
        "name": "tool",
        "value": 3,
    }
    assert safe_state["workspace_dir"] == str(Path("workspaces/test"))

    json.dumps(safe_state)


def test_to_jsonable_can_handle_non_dict_root():
    issues = []

    result = to_jsonable(
        [DummyModel(name="a", value=1)],
        path="$",
        issues=issues,
    )

    assert result == [{"name": "a", "value": 1}]
    assert any(
        issue.code == "PYDANTIC_MODEL_NORMALIZED"
        for issue in issues
    )

    json.dumps(result)