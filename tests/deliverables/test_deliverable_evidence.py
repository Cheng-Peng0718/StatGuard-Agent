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