"""
Tests for progress-gated continuation in the answer-quality gate.

The coverage brief is an LLM and can prescribe evidence that is impossible to
satisfy (e.g. regression_model for two categorical variables). The gate must
NOT loop forever trying to satisfy such a prescription. Continuation is only
recommended when there is missing evidence AND the agent is still making real
progress (substantive analysis run count grew since the last gate visit), with
a one-visit grace period on the first visit.
"""

from __future__ import annotations

from core.deliverables import check_answer_quality


class _FinalAnswer:
    action_type = "final_answer"
    reasoning_summary = "done"


def _chi_run():
    return {
        "tool_name": "run_chi_square", "status": "ok", "success": True,
        "evidence_categories": ["statistical_inference"],
        "metrics": {"p_value": 0.29}, "summary": "chi done",
        "data_version_id": "v1",
    }


def _reg_run():
    return {
        "tool_name": "run_multiple_regression", "status": "ok", "success": True,
        "evidence_categories": ["regression_model"],
        "metrics": {"r_squared": 0.8}, "summary": "reg done",
        "data_version_id": "v1",
    }


def _diag_run():
    return {
        "tool_name": "regression_diagnostics", "status": "ok", "success": True,
        "evidence_categories": ["regression_diagnostics"],
        "metrics": {}, "summary": "diag done", "data_version_id": "v1",
    }


def _brief(required):
    return {
        "analysis_goal": "g",
        "required_evidence_categories": list(required),
        "required_evidence_counts": {c: 1 for c in required},
        "pre_analysis_check_categories": [],
        "optional_evidence_categories": [],
        "autonomy_level": "continue_until_covered",
    }


def _gate(runs, brief, prev):
    return check_answer_quality(
        user_request="x", current_action=_FinalAnswer(),
        analysis_runs=runs, observations=[], active_data_version_id="v1",
        analysis_coverage_brief=brief, prev_substantive_run_count=prev,
    )


class TestProgressGatedContinuation:
    def test_first_visit_grace_allows_one_continuation(self):
        # Missing evidence on first visit -> allowed to continue once.
        c = _gate([_chi_run()], _brief(["regression_model"]), prev=0)
        assert c["missing_evidence_categories"] == ["regression_model"]
        assert c["continuation_recommended"] is True
        assert c["current_substantive_run_count"] == 1

    def test_no_progress_stops_the_loop(self):
        # Second visit, still missing, run count did not grow -> STOP.
        c = _gate([_chi_run()], _brief(["regression_model"]), prev=1)
        assert c["missing_evidence_categories"] == ["regression_model"]
        assert c["continuation_recommended"] is False

    def test_progress_with_remaining_evidence_continues(self):
        # Run count grew (1 -> 2) and something is still missing -> continue.
        c = _gate([_reg_run(), _diag_run()],
                  _brief(["regression_model", "regression_diagnostics", "group_comparison"]),
                  prev=1)
        assert "group_comparison" in c["missing_evidence_categories"]
        assert c["current_substantive_run_count"] == 2
        assert c["continuation_recommended"] is True

    def test_all_covered_stops(self):
        # Nothing missing -> no continuation regardless of progress.
        c = _gate([_reg_run(), _diag_run()],
                  _brief(["regression_model", "regression_diagnostics"]),
                  prev=1)
        assert c["missing_evidence_categories"] == []
        assert c["continuation_recommended"] is False

    def test_blocked_runs_do_not_count_as_progress(self):
        # A blocked regression attempt is not substantive; count stays at 1,
        # so a second visit with only the chi run + a blocked attempt stops.
        blocked = {
            "tool_name": "run_multiple_regression", "status": "blocked",
            "success": False, "evidence_categories": ["regression_model"],
            "summary": "blocked", "data_version_id": "v1",
        }
        c = _gate([_chi_run(), blocked], _brief(["regression_model"]), prev=1)
        assert c["current_substantive_run_count"] == 1  # blocked not counted
        assert c["continuation_recommended"] is False