from types import SimpleNamespace

from core.graph import summarize_node


def make_action(
    *,
    action_id="act_1",
    tool_name="get_summary_stats",
    arguments=None,
):
    return SimpleNamespace(
        action_id=action_id,
        action_type="tool_call",
        tool_name=tool_name,
        arguments=arguments or {},
    )


def test_summarize_node_attaches_state_serialization_audit():
    state = {
        "current_action": make_action(
            action_id="act_summary",
            tool_name="get_summary_stats",
            arguments={},
        ),
        "current_execution": {
            "execution_id": "exec_summary",
            "status": "ok",
            "success": True,
            "error_code": None,
            "message": "Summary completed.",
            "artifacts": [],
            "payload": {
                "n_rows": 5,
            },
        },
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
    }

    result = summarize_node(state)

    assert "execution_audit" in result
    assert "state_serialization_audit" in result

    audit = result["state_serialization_audit"]

    assert audit["status"] in {"ok", "warning", "error"}
    assert "n_issues" in audit
    assert "issues" in audit
    assert "safe_state" not in audit