import os
from verifiers.validators import verify
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from core.state import GraphState
from core.schema import Observation
from core.context_builder import build_context, generate_profile
from core.analysis_tool_plugins.execution import execute_analysis_tool
from core.analysis_runs import build_analysis_run_from_observation
from core.data_versions import (
    get_active_data_path,
    extract_data_version_update,
    validate_data_version_update,
)
from core.dataset_intelligence.profiler import profile_dataframe, summarize_profile
from core.dataset_intelligence.capability_map import build_capability_map

from core.responses import make_assistant_response, make_response_update
from core.planning.execution_queue import mark_plan_step_after_execution

from core.action_access import (
    get_action_arguments,
    get_action_id,
    get_action_tool_name,
    has_action_tool_name,
)

from core.action_codec import action_to_state_dict
from core.verification_codec import verification_to_state_dict

from core.verification_access import (
    get_verification_details,
    get_verification_error_code,
    get_verification_feedback,
    get_verification_status,
    set_verification_fields,
)
from core.execution_codec import normalize_execution_view, execution_to_state_dict

from core.workflow.audit_runtime import attach_execution_audit

from core.workflow.repair_runtime import (
    attach_repair_decision,
    attach_repair_after_summarize,
)

from core.workflow.runtime_utils import sanitize_results, get_action_hash

from core.workflow.nodes.interaction import (
    intent_router_node,
    advisory_answer_node,
)

from core.workflow.nodes.planning import plan_only_node

from core.workflow.routes import (
    route_after_intent,
    route_after_execute_pending_plan,
    route_after_supervisor,
    route_after_verify,
    route_after_review,
    route_after_summarize,
    route_after_deliverable_gate,
)

from core.workflow.nodes.plan_execution import execute_pending_plan_node

from core.workflow.nodes.supervisor import supervisor_node

from core.workflow.nodes.finalization import (
    deliverable_gate_node,
    final_response_node,
)

from core.workflow.execution_fingerprints import has_duplicate_executed_action

def _load_dataframe_for_dataset_intelligence(path: str) -> pd.DataFrame:
    """
    Load the active dataset for Dataset Intelligence.

    This is separate from generate_profile(), because generate_profile()
    returns the legacy DatasetProfile, while Dataset Intelligence needs
    the actual DataFrame.
    """
    lower_path = str(path).lower()

    if lower_path.endswith(".parquet"):
        return pd.read_parquet(path)

    if lower_path.endswith(".csv"):
        return pd.read_csv(path)

    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported active data file type for profiling: {path}")

# --- Graph nodes ---
def build_context_node(state: GraphState):
    step = state.get("current_step", 0) + 1

    current_workspace = state["workspace_dir"]

    # Resolve active data file from the data-version system.
    current_data_path = get_active_data_path(
        workspace_dir=current_workspace,
        data_versions=state.get("data_versions", []) or [],
        active_data_version_id=state.get("active_data_version_id"),
        fallback_file="working_data.parquet",
    )

    if not current_data_path or not os.path.exists(current_data_path):
        raise FileNotFoundError(
            f"No active data file found. "
            f"workspace={current_workspace}, "
            f"active_data_version_id={state.get('active_data_version_id')}, "
            f"resolved_path={current_data_path}"
        )

    # Legacy profile used by the existing supervisor / verifier path.
    # Keep this for compatibility.
    new_profile = generate_profile(current_data_path)

    # New Dataset Intelligence profile.
    # This is the real dataset overview used by advisory / plan-only flows.
    df = _load_dataframe_for_dataset_intelligence(current_data_path)

    active_data_version_id = state.get("active_data_version_id") or "unknown"

    dataset_profile_v2 = profile_dataframe(
        df,
        dataset_name=state.get("dataset_name", "uploaded_dataset"),
        data_version_id=active_data_version_id,
    )

    dataset_summary = summarize_profile(dataset_profile_v2)
    capability_map = build_capability_map(dataset_profile_v2)

    context = build_context(
        step=step,
        max_steps=state["max_steps"],
        user_request=state["user_request"],
        profile=new_profile,
        observations=state.get("observations", []),
        workspace_dir=state.get("workspace_dir", "./"),
        deliverable_check=state.get("deliverable_check"),
        data_versions=state.get("data_versions", []),
        active_data_version_id=state.get("active_data_version_id"),
        data_audit_log=state.get("data_audit_log", []),
    )

    return {
        "current_step": step,
        "current_context_text": context.context_text,

        # Keep legacy profile for current runtime compatibility.
        "dataset_profile": new_profile,

        # New Dataset Intelligence state.
        # Store as plain dict to avoid future LangGraph checkpoint issues.
        "dataset_profile_v2": dataset_profile_v2.model_dump(),
        "dataset_summary": dataset_summary.model_dump(),
        "capability_map": capability_map.model_dump(),
    }

def verify_node(state: GraphState):
    """
    Verification node.

    New contract:
    verify() returns a full VerificationResult produced from
    analysis_tool_plugins.validation.
    """
    action = state["current_action"]

    status, feedback, verify_result = verify(action, state["dataset_profile"])

    tool_name = get_action_tool_name(action)
    arguments = get_action_arguments(action)
    action_id = get_action_id(action)

    action_hash = get_action_hash(tool_name, arguments)

    verification_details = get_verification_details(verify_result)
    verification_details["action_hash"] = action_hash
    verify_result = set_verification_fields(
        verify_result,
        details=verification_details,
    )

    verify_status = get_verification_status(verify_result)
    verify_feedback = get_verification_feedback(verify_result)
    verify_error_code = get_verification_error_code(verify_result)
    verify_details = get_verification_details(verify_result)

    print("\n" + "=" * 40)
    print("[VERIFY NODE DEBUG]")
    print(f"tool_name = {tool_name}")
    print(f"verify_result.status = {verify_status}")
    print(f"verify_result.error_code = {verify_error_code}")
    print(f"verify_result.feedback = {verify_feedback}")
    print(f"verify_result.details = {verify_details}")
    print("=" * 40 + "\n")

    if verify_status in ["rejected_recoverable", "rejected_terminal"]:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action_id,
            tool_name=tool_name,
            arguments=getattr(action, "arguments", {}) or {},
            status="rejected",
            success=False,
            error_code=verify_error_code or "VERIFICATION_FAILED",
            message=verify_feedback,
            artifacts=[],
            summary=(
                f"Validation failed for {tool_name}: "
                f"{verify_feedback}"
            ),
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": verify_error_code or "VERIFICATION_FAILED",
                "message": verify_feedback,
                "details": verify_details,
            },
            raw_data={
                "verification": (
                    verify_result.model_dump()
                    if hasattr(verify_result, "model_dump")
                    else verify_result
                ),
                "recoverable": verify_status == "rejected_recoverable",
            },
        )

        updates = {
            "current_verification": verify_result,
            "observations": [obs.model_dump()],
        }

        # Phase 4E:
        # Verification rejection is also a repair-decision input.
        # Attach repair metadata before clearing current_action, because
        # repair classification needs the source action/tool.
        clear_after_repair = {}

        current_plan_step_id = state.get("current_plan_step_id")
        pending_plan = state.get("pending_plan")

        if current_plan_step_id and pending_plan:
            updated_plan = mark_plan_step_after_execution(
                pending_plan,
                step_id=current_plan_step_id,
                success=False,
                execution_id=None,
                message=verify_feedback,
            )

            content = (
                "I tried to execute the next ready step in the pending plan, "
                "but it failed validation before execution.\n\n"
                f"Tool: {tool_name}\n"
                f"Reason: {verify_feedback}\n\n"
                "I marked this plan step as failed and stopped execution to avoid a retry loop."
            )

            updates.update({
                "pending_plan": updated_plan,
                "plan_status": updated_plan.get("status"),
                "plan_execution_status": "step_verification_failed",
                "assistant_response": make_assistant_response(
                    response_type="error",
                    content=content,
                    source_node="verify",
                    data_version_id=state.get("active_data_version_id"),
                    plan_id=updated_plan.get("plan_id"),
                    plan_status=updated_plan.get("status"),
                    metadata={
                        "error_code": verify_error_code,
                        "tool_name": tool_name,
                        "step_id": current_plan_step_id,
                    },
                ),
            })

            clear_after_repair = {
                "current_plan_step_id": None,
                "current_action": None,
                "current_execution": None,
            }

        updates = attach_repair_decision(state, updates)
        updates.update(clear_after_repair)

        return updates

    updates = {
        "current_verification": verify_result,
        "human_review_required": False,
    }

    return attach_repair_decision(state, updates)

def human_review_node(state: GraphState):
    """
    Phase 0.5 human review node.

    This node does NOT execute the pending action.
    It only records that human confirmation is required.
    """
    vr = state.get("current_verification")
    action = state.get("current_action")

    if vr is None or action is None:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id="unknown",
            tool_name=None,
            arguments={},
            status="rejected",
            success=False,
            error_code="MISSING_REVIEW_STATE",
            message="Human review node was reached without verification or action.",
            artifacts=[],
            summary="Human review could not proceed because verification/action state was missing.",
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": "MISSING_REVIEW_STATE",
            },
            raw_data={},
        )
        return {"observations": [obs.model_dump()]}

    tool_name = get_action_tool_name(action)
    arguments = get_action_arguments(action)

    vr_details = get_verification_details(vr)
    vr_status = get_verification_status(vr)
    feedback = get_verification_feedback(vr)

    action_id = get_action_id(action)
    action_payload = action_to_state_dict(action)
    verification_payload = verification_to_state_dict(vr)

    canonical_arguments = vr_details.get("canonical_arguments") or arguments

    human_review_decision = state.get("human_review_decision")

    submitted_action_hash = state.get("human_review_action_hash")
    expected_action_hash = vr_details.get("action_hash")

    if (
        human_review_decision in {"approved", "rejected"}
        and submitted_action_hash
        and expected_action_hash
        and submitted_action_hash != expected_action_hash
    ):
        content = (
            "The human-review decision did not match the current pending action. "
            "This can happen if the UI was stale or the action changed before approval. "
            "No tool was executed. Please review the current action again."
        )

        return {
            "human_review_required": True,
            "pending_action": action_payload,
            "current_action": action,
            "current_verification": vr,
            "human_review_decision": None,
            "human_review_rejection_reason": None,
            "assistant_response": make_assistant_response(
                response_type="error",
                content=content,
                source_node="human_review",
                data_version_id=state.get("active_data_version_id"),
                metadata={
                    "error_code": "HUMAN_REVIEW_ACTION_HASH_MISMATCH",
                    "submitted_action_hash": submitted_action_hash,
                    "expected_action_hash": expected_action_hash,
                    "tool_name": tool_name,
                    "action_id": action_id,
                },
            ),
        }

    if vr_status == "needs_review" and human_review_decision == "rejected":
        reason = (
            state.get("human_review_rejection_reason")
            or "User rejected the human-review action."
        )

        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action_id,
            tool_name=tool_name,
            arguments=canonical_arguments,
            status="rejected",
            success=False,
            error_code="HUMAN_REVIEW_REJECTED",
            message=reason,
            artifacts=[],
            summary=(
                f"Human review rejected action {tool_name}. "
                f"Reason: {reason}"
            ),
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": "HUMAN_REVIEW_REJECTED",
                "message": reason,
                "pending_action": action_payload or {},
            },
            raw_data={
                "verification": verification_payload or {},
                "pending_action": action_payload or {},
                "human_review_decision": "rejected",
            },
        )

        return {
            "human_review_required": False,
            "pending_action": None,
            "human_review_decision": None,
            "human_review_rejection_reason": None,
            "current_action": None,
            "current_verification": None,
            "observations": [obs.model_dump()],
        }

    # Case -1: backend/user explicitly approved a needs_review action.
    # This is the approval bridge for mutating tools such as clean_data.
    # Do NOT create a rejection observation here, because execute_node's
    # fingerprint gate treats prior observations as executed/attempted actions.
    if vr_status == "needs_review" and human_review_decision == "approved":
        print("[HUMAN REVIEW] User approved needs_review action; routing to execute.")

        approved_feedback = (
            "Human review approved this action for execution."
        )


        if isinstance(vr, dict):
            approved_vr = {
                **vr,
                "status": "allowed",
                "feedback": approved_feedback,
            }

        elif hasattr(vr, "model_copy"):
            approved_vr = vr.model_copy(
                update={
                    "status": "allowed",
                    "feedback": approved_feedback,
                }
            )

        else:
            approved_vr = set_verification_fields(
                vr,
                status="allowed",
                feedback=approved_feedback,
            )

        return {
            "current_verification": approved_vr,
            "human_review_required": False,
            "pending_action": None,
            "human_review_decision": None,
            "current_action": action,
        }

    # Case 0: user approved the pending action.
    # Because the graph was interrupted before human_review,
    # after approval it resumes here first. We do not create an observation here.
    # Routing after human_review will send it to execute.
    if vr_status == "allowed":
        print("[HUMAN REVIEW] User approved action; routing to execute.")
        return {}

    # Case 1: high-risk tool needs user confirmation
    # Case 1: high-risk tool needs user confirmation.
    # Waiting for confirmation is runtime review state, not an execution/rejection observation.
    if vr_status == "needs_review":
        return {
            "human_review_required": True,
            "pending_action": action_payload,
            "current_action": action,
            "current_verification": vr,
        }

    # Case 2: rejected by verifier
    if vr_status in {"rejected_recoverable", "rejected_terminal"}:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action_id,
            tool_name=tool_name,
            arguments=arguments,
            status="rejected",
            success=False,
            error_code="VERIFICATION_FAILED",
            message=feedback,
            artifacts=[],
            summary=f"Action {tool_name} was rejected by verifier: {feedback}",
            structured_data={
                "status": vr_status,
                "success": False,
                "error_code": "VERIFICATION_FAILED",
                "message": feedback,
            },
            raw_data={
                "verification": verification_payload or {},
            },
        )

        return {"observations": [obs.model_dump()]}


    # Safety fallback
    obs = Observation(
        observation_id=f"obs_{uuid.uuid4().hex[:8]}",
        source_action_id=action_id,
        tool_name=tool_name,
        arguments=arguments,
        status="rejected",
        success=False,
        error_code="UNHANDLED_HUMAN_REVIEW_STATUS",
        message=f"Unhandled verification status in human_review_node: {vr_status}",
        artifacts=[],
        summary=f"Unhandled human review status: {vr_status}. Tool was not executed.",
        structured_data={
            "status": vr_status,
            "success": False,
            "error_code": "UNHANDLED_HUMAN_REVIEW_STATUS",
        },
        raw_data={
            "verification": verification_payload or {},
        },
    )

    return {"observations": [obs.model_dump()]}


def execute_node(state: GraphState):
    action = state.get("current_action")
    tool_name = get_action_tool_name(action)
    arguments = get_action_arguments(action)

    if not action or not tool_name:
        message = "Error: No valid action provided."

        return {
            "current_execution": execution_to_state_dict(
                {
                    "status": "blocked",
                    "success": False,
                    "error_code": "NO_VALID_ACTION",
                    "message": message,
                    "artifacts": [],
                    "payload": {},
                },
                fallback_action_id=get_action_id(action),
                fallback_tool_name=tool_name or "unknown_tool",
            )
        }

    if has_duplicate_executed_action(
        state=state,
        tool_name=tool_name,
        arguments=arguments,
    ):
        current_hash = get_action_hash(tool_name, arguments)

        error_msg = (
            f"[System intervention]: Execution refused. You are calling '{tool_name}' "
            f"with parameters identical to a previous executed attempt.\n"
            f"To retry, change arguments or explicitly choose a different strategy."
        )

        print(
            f"[Fingerprint gate]: blocked duplicate executed action "
            f"{tool_name} (fp: {current_hash[:6]})"
        )

        return {
            "current_execution": execution_to_state_dict(
                {
                    "status": "blocked",
                    "success": False,
                    "error_code": "DUPLICATE_EXECUTION_ATTEMPT",
                    "message": error_msg,
                    "artifacts": [],
                    "payload": {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "action_hash": current_hash,
                    },
                },
                fallback_action_id=get_action_id(action),
                fallback_tool_name=tool_name,
            )
        }

    print(f"[Execute]: {tool_name}")

    context_pkg = build_context(
        step=state.get("current_step", 1),
        max_steps=state.get("max_steps", 20),
        user_request=state.get("user_request", "Not provided"),
        profile=state.get("dataset_profile"),
        observations=state.get("observations", []),
        workspace_dir=state.get("workspace_dir", "./"),
        deliverable_check=state.get("deliverable_check"),
        data_versions=state.get("data_versions", []),
        active_data_version_id=state.get("active_data_version_id"),
        data_audit_log=state.get("data_audit_log", []),
    )

    exec_result = execute_analysis_tool(action, context_pkg)

    if hasattr(exec_result, "model_dump"):
        raw_payload = exec_result.model_dump()
    elif hasattr(exec_result, "dict"):
        raw_payload = exec_result.dict()
    else:
        raw_payload = exec_result

    safe_result = sanitize_results(raw_payload)

    return {
        "current_execution": execution_to_state_dict(
            safe_result,
            fallback_action_id=get_action_id(action),
            fallback_tool_name=tool_name,
        )
    }

def summarize_node(state: GraphState):
    current_action = state.get("current_action")
    tool_name = get_action_tool_name(current_action, "unknown_tool")
    arguments = get_action_arguments(current_action)

    raw_result = state.get("current_execution", "No execution result")

    execution_view = normalize_execution_view(
        raw_result,
        fallback_action_id=get_action_id(current_action),
        fallback_tool_name=tool_name,
    )

    execution_id = execution_view.get("execution_id")
    status = execution_view.get("status")
    success = bool(execution_view.get("success"))
    error_code = execution_view.get("error_code")
    message = execution_view.get("message")
    artifacts = execution_view.get("artifacts") or []
    payload = execution_view.get("payload") or {}
    raw_result = execution_view

    summary = (
        f"Tool {tool_name} finished with status={status}, success={success}. "
        f"message={message or 'No message'}"
    )

    if error_code:
        summary += f" error_code={error_code}."

    refined_observation = {
        "observation_id": f"obs_{uuid.uuid4().hex[:8]}",
        "source_action_id": get_action_id(current_action),
        "tool_name": tool_name,
        "arguments": arguments,

        # Phase 2: provenance
        "data_version_id": state.get("active_data_version_id"),

        "status": status,
        "success": success,
        "error_code": error_code,
        "message": message,
        "artifacts": artifacts,
        "summary": summary,
        "structured_data": {
            "status": status,
            "success": success,
            "error_code": error_code,
            "message": message,
            "artifacts": artifacts,
            "payload": payload,
            # Phase 2: provenance
            "data_version_id": state.get("active_data_version_id"),
        },
        "raw_data": raw_result,
    }

    print(f"[Summarize]: archived result for {tool_name}.")

    # Base graph-state updates for every executed action.
    # IMPORTANT:
    # Define updates BEFORE applying data_version_update.
    updates = {
        "observations": [refined_observation],

        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,

        "current_step": state.get("current_step", 0) + 1,
    }

    # Extract data_version_update from the new plugin execution protocol.
    # Canonical path:
    # ToolExecutionResult.payload["data_version_update"]
    data_version_update = extract_data_version_update(raw_result)
    validated_version_update = validate_data_version_update(data_version_update)

    if validated_version_update is not None:
        new_version = validated_version_update["new_version"]
        new_active_id = validated_version_update["active_data_version_id"]
        audit_event = validated_version_update.get("audit_event")

        existing_versions = state.get("data_versions", []) or []
        existing_audit_log = state.get("data_audit_log", []) or []

        updates["data_versions"] = existing_versions + [new_version]
        updates["active_data_version_id"] = new_active_id

        if audit_event:
            updates["data_audit_log"] = existing_audit_log + [audit_event]

        # Mutating tools should make the observation point to the new active version.
        refined_observation["data_version_id"] = new_active_id

        structured_data = refined_observation.get("structured_data")
        if isinstance(structured_data, dict):
            structured_data["data_version_id"] = new_active_id

        # Also keep payload provenance aligned.
        if isinstance(payload, dict):
            payload["active_data_version_id"] = new_active_id
            payload["data_version_id"] = new_active_id

        print(f"[DATA VERSION] active_data_version_id -> {new_active_id}")

    # Phase 3: append every real tool execution to Analysis Results registry.
    # Put this AFTER data_version_update so mutating tools record the new data version.
    #
    # S12C:
    # AnalysisRun must exist for both successful and failed tool executions.
    # Failed runs are required evidence for DeliverableGate and future repair logic.
    if tool_name not in {"unknown_tool"}:
        analysis_run = build_analysis_run_from_observation(
            observation=refined_observation,
        )

        updates["analysis_runs"] = [analysis_run]


    # Phase 4: if this execution came from a pending plan step,
    # mark that PlanStep as completed or failed.
    current_plan_step_id = state.get("current_plan_step_id")
    pending_plan = state.get("pending_plan")

    if current_plan_step_id and pending_plan:
        updated_plan = mark_plan_step_after_execution(
            pending_plan,
            step_id=current_plan_step_id,
            success=success,
            execution_id=execution_id,
            message=message,
        )

        updates["pending_plan"] = updated_plan
        updates["plan_status"] = updated_plan.get("status")
        updates["current_plan_step_id"] = None

        # S4: after a pending-plan step is summarized,
        # clear the action origin. Routing will still see the previous state,
        # but subsequent state should not keep stale origin.
        updates["action_origin"] = None

        print(
            f"[PLAN EXECUTION] step {current_plan_step_id} "
            f"marked as {'completed' if success else 'failed'}"
        )

    updates = attach_repair_after_summarize(
        state=state,
        updates=updates,
        current_action=current_action,
        raw_result=raw_result,
        tool_name=tool_name,
    )

    return attach_execution_audit(state, updates)

# --- Compile graph ---
workflow = StateGraph(GraphState)

workflow.add_node("build_context", build_context_node)
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("advisory_answer", advisory_answer_node)
workflow.add_node("plan_only", plan_only_node)
workflow.add_node("execute_pending_plan", execute_pending_plan_node)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("verify", verify_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("execute", execute_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("deliverable_gate", deliverable_gate_node)
workflow.add_node("final_response", final_response_node)

workflow.set_entry_point("build_context")

# Step 1: build context, dataset profile, dataset summary, capability map.
workflow.add_edge("build_context", "intent_router")

# Step 2: route by interaction intent.
workflow.add_conditional_edges(
    "intent_router",
    route_after_intent,
    {
        "advisory_answer": "advisory_answer",
        "plan_only": "plan_only",
        "execute_pending_plan": "execute_pending_plan",
        "supervisor": "supervisor",
    },
)

# These modes must never execute tools.
workflow.add_edge("advisory_answer", END)
workflow.add_edge("plan_only", END)

workflow.add_conditional_edges(
    "execute_pending_plan",
    route_after_execute_pending_plan,
    {
        "verify": "verify",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "verify": "verify",
        "deliverable_gate": "deliverable_gate",
        "end": END,
    }
)

workflow.add_conditional_edges(
    "deliverable_gate",
    route_after_deliverable_gate,
    {
        "final_response": "final_response",
        "build_context": "build_context",
        "end": END,
    },
)

workflow.add_edge("final_response", END)

workflow.add_conditional_edges(
    "verify",
    route_after_verify,
    {
        "execute": "execute",
        "human_review": "human_review",
        "build_context": "build_context",
    },
)

workflow.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "execute": "execute",
        "build_context": "build_context",
    },
)

workflow.add_edge("execute", "summarize")

workflow.add_conditional_edges(
    "summarize",
    route_after_summarize,
    {
        "build_context": "build_context",
        "end": END,
    },
)



def create_graph_app():
    """
    Build the compiled LangGraph app explicitly.

    Importing core.graph should expose node functions and workflow wiring,
    but should not compile a global runnable app as an import-time side effect.
    """
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],
    )