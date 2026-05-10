from __future__ import annotations

import importlib
import uuid


def _compile_graph_with_patched_nodes(monkeypatch, *, nodes):
    import core.workflow.nodes.context as context_node
    import core.workflow.nodes.execution as execution_node
    import core.workflow.nodes.finalization as finalization_node
    import core.workflow.nodes.human_review as human_review_node
    import core.workflow.nodes.interaction as interaction_node
    import core.workflow.nodes.plan_execution as plan_execution_node
    import core.workflow.nodes.planning as planning_node
    import core.workflow.nodes.summarization as summarization_node
    import core.workflow.nodes.supervisor as supervisor_node
    import core.workflow.nodes.verification as verification_node

    module_by_node = {
        "build_context": (context_node, "build_context_node"),
        "intent_router": (interaction_node, "intent_router_node"),
        "advisory_answer": (interaction_node, "advisory_answer_node"),
        "plan_only": (planning_node, "plan_only_node"),
        "execute_pending_plan": (plan_execution_node, "execute_pending_plan_node"),
        "supervisor": (supervisor_node, "supervisor_node"),
        "verify": (verification_node, "verify_node"),
        "human_review": (human_review_node, "human_review_node"),
        "execute": (execution_node, "execute_node"),
        "summarize": (summarization_node, "summarize_node"),
        "deliverable_gate": (finalization_node, "deliverable_gate_node"),
        "final_response": (finalization_node, "final_response_node"),
    }

    for node_name, replacement in nodes.items():
        module, attr_name = module_by_node[node_name]
        monkeypatch.setattr(module, attr_name, replacement)

    import core.graph as graph_module

    graph_module = importlib.reload(graph_module)
    return graph_module.create_graph_app()


def _invoke(app, state):
    return app.invoke(
        state,
        config={"configurable": {"thread_id": f"graph-kernel-{uuid.uuid4().hex}"}},
    )


def _base_state(user_request: str) -> dict:
    return {
        "user_request": user_request,
        "workspace_dir": "./tmp",
        "current_step": 0,
        "max_steps": 2,
        "observations": [],
        "analysis_runs": [],
        "data_versions": [],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,
        "task_contract": None,
        "deliverable_check": None,
        "repair_attempts": [],
    }


def _build_context_update(state):
    return {
        "current_step": state.get("current_step", 0) + 1,
        "dataset_profile": {"columns": {}},
        "dataset_profile_v2": {
            "dataset_name": "kernel_fixture",
            "data_version_id": "raw_v1",
            "n_rows": 5,
            "n_cols": 2,
            "columns": {},
        },
        "dataset_summary": {
            "n_rows": 5,
            "n_cols": 2,
            "numeric_columns": ["GPA", "SATM"],
            "categorical_columns": [],
            "binary_columns": [],
            "id_like_columns": [],
            "missingness_summary": {"n_columns_with_missing": 0},
        },
        "capability_map": {"capabilities": []},
        "current_context_text": "graph kernel test context",
    }


def test_graph_kernel_plan_generation_flow_does_not_execute_tools(monkeypatch):
    def fake_intent_router(state):
        return {
            "interaction_intent": "plan_only",
            "intent_decision": {
                "intent": "plan_analysis",
                "confidence": 0.9,
                "reason": "User asked for a plan.",
                "should_execute": False,
            },
            "task_spec": {
                "goal_type": "dataset_overview",
                "user_goal": "Understand the dataset.",
            },
        }

    def fake_plan_only(state):
        return {
            "pending_plan": {
                "plan_id": "plan_graph_kernel_overview",
                "status": "verified",
                "steps": [
                    {"tool_name": "inspect_dataset"},
                    {"tool_name": "missingness_report"},
                    {"tool_name": "get_summary_stats"},
                ],
                "blocked_or_not_recommended": [],
            },
            "plan_status": "verified",
            "assistant_response": {
                "response_type": "plan",
                "content": "Dataset overview plan.",
                "source_node": "plan_only",
            },
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
        }

    app = _compile_graph_with_patched_nodes(
        monkeypatch,
        nodes={
            "build_context": _build_context_update,
            "intent_router": fake_intent_router,
            "plan_only": fake_plan_only,
        },
    )

    result = _invoke(app, _base_state("make a plan for this dataset"))

    step_tools = [
        step["tool_name"]
        for step in result["pending_plan"]["steps"]
    ]

    assert result["assistant_response"]["response_type"] == "plan"
    assert result["plan_status"] == "verified"
    assert "run_multiple_regression" not in step_tools
    assert result["current_action"] is None
    assert result["current_execution"] is None


def test_graph_kernel_direct_regression_flow_records_observation_and_run(monkeypatch):
    def fake_intent_router(state):
        return {
            "interaction_intent": "unknown",
            "intent_decision": {
                "intent": "direct_analysis",
                "confidence": 0.95,
                "reason": "User requested regression.",
                "should_execute": True,
            },
            "task_spec": {
                "goal_type": "regression_modeling",
                "target_variables": ["GPA"],
                "predictor_variables": ["SATM"],
            },
        }

    def fake_supervisor(state):
        return {
            "current_action": {
                "action_id": "act_regression",
                "action_type": "tool_call",
                "tool_name": "run_multiple_regression",
                "arguments": {
                    "target_col": "GPA",
                    "feature_cols": ["SATM"],
                },
                "reasoning_summary": "Run the requested regression.",
            },
            "task_contract": {
                "contract_id": "contract_regression",
                "user_goal": "Fit GPA on SATM.",
                "required_deliverables": [
                    {
                        "deliverable_id": "regression_model",
                        "description": "Fit OLS regression.",
                        "satisfied_by": ["run_multiple_regression"],
                        "required_evidence": ["status_ok", "coef_table", "r_squared"],
                        "status": "pending",
                    }
                ],
                "constraints": [],
                "created_by": "supervisor",
                "status": "active",
            },
        }

    def fake_verify_node(state):
        return {
            "current_verification": {
                "status": "allowed",
                "feedback": "ok",
                "error_code": None,
                "details": {},
            },
            "human_review_required": False,
        }

    def fake_execute(state):
        return {
            "current_execution": {
                "execution_id": "exec_regression",
                "action_id": "act_regression",
                "tool_name": "run_multiple_regression",
                "status": "ok",
                "success": True,
                "payload": {
                    "metrics": {"r_squared": 0.72},
                    "tables": {"coef_table": [{"term": "SATM", "estimate": 0.1}]},
                },
                "artifacts": [],
            }
        }

    def fake_summarize(state):
        return {
            "current_step": state.get("max_steps", 2),
            "observations": [
                {
                    "observation_id": "obs_regression",
                    "tool_name": "run_multiple_regression",
                    "status": "ok",
                    "success": True,
                    "summary": "Regression completed.",
                    "structured_data": {
                        "r_squared": 0.72,
                        "coef_table": [{"term": "SATM", "estimate": 0.1}],
                    },
                }
            ],
            "analysis_runs": [
                {
                    "run_id": "run_regression",
                    "tool_name": "run_multiple_regression",
                    "status": "ok",
                    "success": True,
                    "metrics": {"r_squared": 0.72},
                    "tables": {"coef_table": [{"term": "SATM", "estimate": 0.1}]},
                    "artifacts": [],
                }
            ],
            "execution_audit": {"status": "ok", "issues": []},
        }

    app = _compile_graph_with_patched_nodes(
        monkeypatch,
        nodes={
            "build_context": _build_context_update,
            "intent_router": fake_intent_router,
            "supervisor": fake_supervisor,
            "verify": fake_verify_node,
            "execute": fake_execute,
            "summarize": fake_summarize,
        },
    )

    result = _invoke(app, _base_state("run linear regression of GPA on SATM"))

    assert result["task_spec"]["goal_type"] == "regression_modeling"
    assert result["current_verification"]["status"] == "allowed"
    assert result["current_execution"]["status"] == "ok"
    assert result["observations"][0]["tool_name"] == "run_multiple_regression"
    assert result["analysis_runs"][0]["metrics"]["r_squared"] == 0.72
    assert result["task_contract"]["contract_id"] == "contract_regression"


def test_graph_kernel_clean_data_requires_review_before_execution(monkeypatch):
    def fake_intent_router(state):
        return {
            "interaction_intent": "unknown",
            "intent_decision": {
                "intent": "direct_analysis",
                "confidence": 0.95,
                "reason": "User requested data mutation.",
                "should_execute": True,
            },
            "task_spec": {
                "goal_type": "data_cleaning",
                "target_variables": ["GPA"],
            },
        }

    def fake_supervisor(state):
        return {
            "current_action": {
                "action_id": "act_clean",
                "action_type": "tool_call",
                "tool_name": "clean_data",
                "arguments": {
                    "action_type": "drop",
                    "strategy": "rows",
                    "columns": ["GPA"],
                },
                "reasoning_summary": "Drop rows with missing GPA.",
            },
        }

    def fake_validate_plugin_action(state):
        return {
            "current_verification": {
                "action_id": "act_clean",
                "status": "needs_review",
                "feedback": "clean_data mutates data and requires confirmation.",
                "details": {"requires_confirmation": True},
            },
            "human_review_required": True,
            "pending_action": state["current_action"],
        }

    def fail_if_executed(state):
        raise AssertionError("clean_data executed before human review")

    app = _compile_graph_with_patched_nodes(
        monkeypatch,
        nodes={
            "build_context": _build_context_update,
            "intent_router": fake_intent_router,
            "supervisor": fake_supervisor,
            "verify": fake_validate_plugin_action,
            "human_review": fail_if_executed,
            "execute": fail_if_executed,
        },
    )

    result = _invoke(app, _base_state("drop rows with missing GPA"))

    assert result["current_verification"]["status"] == "needs_review"
    assert result["human_review_required"] is True
    assert result["pending_action"]["tool_name"] == "clean_data"
    assert result["current_execution"] is None
    assert result["observations"] == []
    assert result["analysis_runs"] == []


def test_graph_kernel_deliverable_contract_flow_reaches_final_response(monkeypatch):
    def fake_intent_router(state):
        return {
            "interaction_intent": "unknown",
            "intent_decision": {
                "intent": "direct_analysis",
                "confidence": 0.95,
                "reason": "User requested final regression report.",
                "should_execute": True,
            },
            "task_spec": {
                "goal_type": "regression_modeling",
                "target_variables": ["GPA"],
                "predictor_variables": ["SATM"],
            },
        }

    def fake_supervisor(state):
        return {
            "current_action": {
                "action_id": "act_final",
                "action_type": "final_answer",
                "tool_name": "none",
                "arguments": {
                    "final_answer": "The regression model was fit and reviewed."
                },
                "reasoning_summary": "All required evidence is available.",
            },
            "task_contract": {
                "contract_id": "contract_regression",
                "user_goal": "Fit GPA on SATM.",
                "required_deliverables": [
                    {
                        "deliverable_id": "regression_model",
                        "description": "Fit OLS regression.",
                        "satisfied_by": ["run_multiple_regression"],
                        "required_evidence": ["status_ok", "coef_table", "r_squared"],
                        "status": "pending",
                    }
                ],
                "constraints": [],
                "created_by": "supervisor",
                "status": "active",
            },
        }

    app = _compile_graph_with_patched_nodes(
        monkeypatch,
        nodes={
            "build_context": _build_context_update,
            "intent_router": fake_intent_router,
            "supervisor": fake_supervisor,
        },
    )

    state = _base_state("finish the regression report")
    state["analysis_runs"] = [
        {
            "run_id": "run_regression",
            "tool_name": "run_multiple_regression",
            "status": "ok",
            "success": True,
            "metrics": {"r_squared": 0.72},
            "tables": {"coef_table": [{"term": "SATM", "estimate": 0.1}]},
            "artifacts": [],
        }
    ]
    state["execution_audit"] = {"status": "ok", "issues": []}

    result = _invoke(app, state)

    assert result["deliverable_check"]["status"] == "ok"
    assert "deliverable:regression_model" in result["deliverable_check"]["satisfied"]
    assert "criterion:evidence:coef_table" in result["deliverable_check"]["satisfied"]
    assert result["assistant_response"]["response_type"] == "final_answer"
    assert result["assistant_response"]["content"] == (
        "The regression model was fit and reviewed."
    )
    assert result["assistant_response"]["metadata"]["deliverable_status"] == "ok"
