from core.deliverables import check_answer_quality


class DummyAction:
    def __init__(self, action_type="final_answer"):
        self.action_type = action_type


def test_answer_quality_gate_passes_with_substantive_current_analysis_run():
    check = check_answer_quality(
        user_request="Run a regression.",
        current_action=DummyAction("final_answer"),
        active_data_version_id="data_v_1",
        observations=[],
        analysis_runs=[
            {
                "tool_name": "run_multiple_regression",
                "title": "Linear Model",
                "status": "ok",
                "data_version_id": "data_v_1",
                "summary": "Fitted a regression model.",
                "metrics": {
                    "r_squared": 0.61,
                    "f_p_value": 0.00001,
                },
                "tables": {
                    "coef_table": [],
                    "assumptions_and_limitations": [],
                },
                "guardrails": [],
            }
        ],
    )

    assert check["status"] == "ok"
    assert check["gate_type"] == "answer_quality_gate"
    assert check["quality_status"] == "pass"
    assert not check["warnings"]


def test_answer_quality_gate_warns_when_only_data_prep_runs_exist():
    check = check_answer_quality(
        user_request="Analyze revenue.",
        current_action=DummyAction("final_answer"),
        active_data_version_id="data_v_1",
        observations=[],
        analysis_runs=[
            {
                "tool_name": "materialize_sql_query_result",
                "title": "Materialize SQL Query Result",
                "status": "ok",
                "data_version_id": "data_v_1",
                "summary": "Materialized a dataset.",
                "metrics": {},
                "tables": {},
                "guardrails": [],
            }
        ],
    )

    assert check["status"] == "ok"
    assert check["gate_type"] == "answer_quality_gate"
    assert check["quality_status"] == "needs_attention"

    warning_ids = {item["check_id"] for item in check["warnings"]}
    assert "substantive_analysis_present" in warning_ids


def test_answer_quality_gate_allows_ask_user_action_without_analysis_runs():
    check = check_answer_quality(
        user_request="Analyze my data.",
        current_action=DummyAction("ask_user"),
        active_data_version_id=None,
        observations=[],
        analysis_runs=[],
    )

    assert check["status"] == "ok"
    assert check["gate_type"] == "answer_quality_gate"
    assert check["quality_status"] == "pass"


def test_answer_quality_gate_warns_on_stale_substantive_run():
    check = check_answer_quality(
        user_request="Run a model.",
        current_action=DummyAction("final_answer"),
        active_data_version_id="data_v_new",
        observations=[],
        analysis_runs=[
            {
                "tool_name": "run_multiple_regression",
                "title": "Linear Model",
                "status": "ok",
                "data_version_id": "data_v_old",
                "summary": "Fitted a regression model.",
                "metrics": {
                    "r_squared": 0.61,
                },
                "tables": {
                    "coef_table": [],
                    "assumptions_and_limitations": [],
                },
                "guardrails": [],
            }
        ],
    )

    assert check["status"] == "ok"
    assert check["quality_status"] == "needs_attention"

    warning_ids = {item["check_id"] for item in check["warnings"]}
    assert "active_data_version_alignment" in warning_ids


def test_answer_quality_gate_warns_when_statistical_run_has_no_limitations():
    check = check_answer_quality(
        user_request="Run a regression.",
        current_action=DummyAction("final_answer"),
        active_data_version_id="data_v_1",
        observations=[],
        analysis_runs=[
            {
                "tool_name": "run_multiple_regression",
                "title": "Linear Model",
                "status": "ok",
                "data_version_id": "data_v_1",
                "summary": "Fitted a regression model.",
                "metrics": {
                    "r_squared": 0.61,
                    "f_p_value": 0.00001,
                },
                "tables": {
                    "coef_table": [],
                },
                "guardrails": [],
            }
        ],
    )

    assert check["status"] == "ok"
    assert check["quality_status"] == "needs_attention"

    warning_ids = {item["check_id"] for item in check["warnings"]}
    assert "statistical_limitations_present" in warning_ids