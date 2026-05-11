from types import SimpleNamespace

from core.repair.attempts import (
    append_repair_attempt,
    can_attempt_repair,
    count_repair_attempts_for_action,
    make_repair_attempt,
    normalize_repair_attempts,
)


def make_action(
    *,
    action_id="act_1",
    tool_name="clean_data",
    arguments=None,
):
    return SimpleNamespace(
        action_id=action_id,
        action_type="tool_call",
        tool_name=tool_name,
        arguments=arguments or {},
    )


def test_make_repair_attempt_records_decision_and_action():
    action = make_action(
        action_id="act_clean",
        tool_name="clean_data",
    )

    repair_decision = {
        "status": "repairable",
        "reason": "Invalid arguments can be repaired.",
        "tool_name": "clean_data",
        "error_code": "INVALID_TOOL_ARGUMENTS",
    }

    attempt = make_repair_attempt(
        repair_decision=repair_decision,
        current_action=action,
        repair_type="argument_repair",
        proposed_arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA"],
        },
        message="Normalize clean_data strategy.",
    )

    assert attempt["repair_attempt_id"].startswith("repair_")
    assert attempt["source_action_id"] == "act_clean"
    assert attempt["source_tool_name"] == "clean_data"
    assert attempt["decision_status"] == "repairable"
    assert attempt["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert attempt["repair_type"] == "argument_repair"
    assert attempt["proposed_arguments"]["strategy"] == "rows"


def test_append_repair_attempt_preserves_existing_attempts():
    first = {
        "repair_attempt_id": "repair_1",
        "source_action_id": "act_1",
    }

    second = {
        "repair_attempt_id": "repair_2",
        "source_action_id": "act_1",
    }

    result = append_repair_attempt([first], second)

    assert result == [first, second]


def test_count_repair_attempts_for_action():
    attempts = [
        {
            "repair_attempt_id": "repair_1",
            "source_action_id": "act_1",
        },
        {
            "repair_attempt_id": "repair_2",
            "source_action_id": "act_1",
        },
        {
            "repair_attempt_id": "repair_3",
            "source_action_id": "act_2",
        },
    ]

    assert count_repair_attempts_for_action(
        attempts,
        source_action_id="act_1",
    ) == 2


def test_can_attempt_repair_true_when_under_policy_limit():
    action = make_action(
        action_id="act_clean",
        tool_name="clean_data",
    )

    repair_decision = {
        "status": "repairable",
        "tool_name": "clean_data",
        "error_code": "INVALID_TOOL_ARGUMENTS",
    }

    assert can_attempt_repair(
        repair_decision=repair_decision,
        repair_attempts=[],
        current_action=action,
    ) is True


def test_can_attempt_repair_false_for_terminal_decision():
    action = make_action(
        action_id="act_reg",
        tool_name="run_multiple_regression",
    )

    repair_decision = {
        "status": "terminal",
        "tool_name": "run_multiple_regression",
        "error_code": "INTERNAL_PLUGIN_ERROR",
    }

    assert can_attempt_repair(
        repair_decision=repair_decision,
        repair_attempts=[],
        current_action=action,
    ) is False


def test_normalize_repair_attempts_accepts_log_dict():
    result = normalize_repair_attempts({
        "attempts": [
            {
                "repair_attempt_id": "repair_1",
            }
        ]
    })

    assert result == [
        {
            "repair_attempt_id": "repair_1",
        }
    ]