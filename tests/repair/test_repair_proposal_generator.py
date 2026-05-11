from types import SimpleNamespace

from core.repair.proposal_generator import generate_repair_proposal


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


def test_generate_no_op_proposal_when_no_repair_needed():
    proposal = generate_repair_proposal(
        repair_decision={
            "status": "no_repair_needed",
            "tool_name": "get_summary_stats",
        },
        current_action=make_action(
            tool_name="get_summary_stats",
        ),
    )

    assert proposal["proposal_type"] == "no_op"
    assert proposal["requires_user"] is False


def test_generate_no_op_proposal_for_terminal_decision():
    proposal = generate_repair_proposal(
        repair_decision={
            "status": "terminal",
            "tool_name": "run_multiple_regression",
            "error_code": "INTERNAL_PLUGIN_ERROR",
        },
        current_action=make_action(
            action_id="act_reg",
            tool_name="run_multiple_regression",
        ),
    )

    assert proposal["proposal_type"] == "no_op"
    assert proposal["source_action_id"] == "act_reg"
    assert proposal["source_error_code"] == "INTERNAL_PLUGIN_ERROR"


def test_generate_argument_repair_from_schema_value_aliases_for_clean_data():
    proposal = generate_repair_proposal(
        repair_decision={
            "status": "repairable",
            "tool_name": "clean_data",
            "error_code": "INVALID_TOOL_ARGUMENTS",
        },
        current_action=make_action(
            action_id="act_clean",
            tool_name="clean_data",
            arguments={
                "action_type": "drop rows",
                "strategy": "drop",
                "columns": ["GPA"],
            },
        ),
    )

    assert proposal["proposal_type"] == "argument_repair"
    assert proposal["source_action_id"] == "act_clean"
    assert proposal["source_tool_name"] == "clean_data"
    assert proposal["proposed_tool_name"] == "clean_data"

    assert proposal["proposed_arguments"]["action_type"] == "drop"
    assert proposal["proposed_arguments"]["strategy"] == "rows"
    assert proposal["proposed_arguments"]["columns"] == ["GPA"]


def test_generate_ask_user_proposal_for_missing_columns():
    proposal = generate_repair_proposal(
        repair_decision={
            "status": "needs_user",
            "tool_name": "run_multiple_regression",
            "error_code": "MISSING_COLUMNS",
        },
        current_action=make_action(
            action_id="act_reg",
            tool_name="run_multiple_regression",
            arguments={
                "target_col": "GPA",
            },
        ),
    )

    assert proposal["proposal_type"] == "ask_user"
    assert proposal["requires_user"] is True
    assert proposal["source_action_id"] == "act_reg"
    assert "missing_fields" in proposal["metadata"]


def test_generate_no_op_when_repairable_but_no_deterministic_fix_available():
    proposal = generate_repair_proposal(
        repair_decision={
            "status": "repairable",
            "tool_name": "run_multiple_regression",
            "error_code": "MODEL_FIT_FAILED",
        },
        current_action=make_action(
            action_id="act_reg",
            tool_name="run_multiple_regression",
            arguments={
                "target_col": "GPA",
                "feature_cols": ["SATM"],
            },
        ),
    )

    assert proposal["proposal_type"] in {"no_op", "argument_repair"}

    if proposal["proposal_type"] == "no_op":
        assert "no deterministic" in proposal["reason"].lower()