from core.deliverables.gate import evaluate_deliverable_gate_state


def test_deliverable_gate_allows_when_no_task_contract():
    result = evaluate_deliverable_gate_state({
        "analysis_runs": [],
    })

    assert result.status == "ok"
    assert result.message == "No task_contract declared."


def test_deliverable_gate_blocks_final_answer_when_required_tool_missing():
    result = evaluate_deliverable_gate_state({
        "task_contract": {
            "required_tools": ["get_summary_stats"],
        },
        "analysis_runs": [],
    })

    assert result.status == "needs_more_work"
    assert "tool:get_summary_stats" in result.missing


def test_deliverable_gate_allows_when_required_tool_succeeded():
    result = evaluate_deliverable_gate_state({
        "task_contract": {
            "required_tools": ["get_summary_stats"],
        },
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
                "artifacts": [],
            }
        ],
    })

    assert result.status == "ok"
    assert "tool:get_summary_stats" in result.satisfied
    assert result.missing == []


def test_deliverable_gate_requires_missing_artifact():
    result = evaluate_deliverable_gate_state({
        "task_contract": {
            "required_tools": ["generate_scatterplot"],
            "required_artifacts": ["plot"],
        },
        "analysis_runs": [
            {
                "tool_name": "generate_scatterplot",
                "status": "ok",
                "success": True,
                "artifacts": [],
            }
        ],
    })

    assert result.status == "needs_more_work"
    assert "tool:generate_scatterplot" in result.satisfied
    assert "artifact:plot" in result.missing


def test_deliverable_gate_allows_required_artifact_when_present():
    result = evaluate_deliverable_gate_state({
        "task_contract": {
            "required_tools": ["generate_scatterplot"],
            "required_artifacts": ["plot"],
        },
        "analysis_runs": [
            {
                "tool_name": "generate_scatterplot",
                "status": "ok",
                "success": True,
                "artifacts": [
                    {
                        "artifact_type": "plot",
                        "path": "plots/scatter.png",
                    }
                ],
            }
        ],
    })

    assert result.status == "ok"
    assert "artifact:plot" in result.satisfied


def test_deliverable_gate_flags_failed_required_tool():
    result = evaluate_deliverable_gate_state({
        "task_contract": {
            "required_tools": ["run_multiple_regression"],
        },
        "analysis_runs": [
            {
                "tool_name": "run_multiple_regression",
                "status": "failed",
                "success": False,
                "message": "Model failed.",
            }
        ],
    })

    assert result.status == "needs_more_work"
    assert "tool_failed:run_multiple_regression" in result.blocked

def test_deliverable_gate_blocks_missing_success_criteria():
    result = evaluate_deliverable_gate_state({
        "task_contract": {
            "success_criteria": ["mention limitations"],
        },
        "analysis_runs": [],
    })

    assert result.status == "needs_more_work"
    assert "criterion:mention limitations" in result.missing


def test_deliverable_gate_allows_partial_missing_when_allow_partial_true():
    result = evaluate_deliverable_gate_state({
        "task_contract": {
            "required_tools": ["get_summary_stats"],
            "required_deliverables": ["brief summary"],
            "allow_partial": True,
        },
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
            }
        ],
    })

    assert result.status == "ok"
    assert "deliverable:brief summary" in result.missing
    assert "allow_partial=True" in result.message

from core.deliverables.evidence import (
    extract_final_answer_content_from_state,
    criterion_satisfied_by_final_answer_text,
    get_satisfied_criterion_names,
    get_satisfied_deliverable_names,
)


def test_extract_final_answer_from_state_direct_field():
    assert extract_final_answer_content_from_state({
        "final_answer": "This is final."
    }) == "This is final."


def test_extract_final_answer_from_current_action_arguments():
    assert extract_final_answer_content_from_state({
        "current_action": {
            "arguments": {
                "final_answer": "Answer from arguments."
            }
        }
    }) == "Answer from arguments."


def test_contains_criterion_satisfied_by_final_answer_text():
    assert criterion_satisfied_by_final_answer_text(
        "contains:limitations",
        "This analysis has several limitations."
    ) is True


def test_freeform_criterion_not_satisfied_by_text_in_s10c():
    assert criterion_satisfied_by_final_answer_text(
        "mention limitations",
        "This analysis has several limitations."
    ) is False


def test_get_explicit_satisfied_deliverables_and_criteria():
    state = {
        "deliverable_evidence": {
            "satisfied_deliverables": ["brief summary"],
            "satisfied_criteria": ["mention limitations"],
            "satisfied": [
                "deliverable:diagnostic paragraph",
                "criterion:contains:p-value",
            ],
        }
    }

    assert "brief summary" in get_satisfied_deliverable_names(state)
    assert "diagnostic paragraph" in get_satisfied_deliverable_names(state)

    assert "mention limitations" in get_satisfied_criterion_names(state)
    assert "contains:p-value" in get_satisfied_criterion_names(state)