import os
from verifiers.validators import verify
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import numpy as np
import pandas as pd
from core.state import GraphState
from core.schema import Observation
from core.context_builder import build_context, generate_profile
from agents.supervisor import call_supervisor
from core.analysis_tool_plugins.execution import execute_analysis_tool
import hashlib
import json
from core.analysis_runs import build_analysis_run_from_observation
from core.data_versions import get_active_data_path
from core.dataset_intelligence.profiler import profile_dataframe, summarize_profile
from core.dataset_intelligence.capability_map import build_capability_map
from core.interaction_intent import classify_interaction_intent
from core.dataset_intelligence.schemas import CapabilityMap, DatasetProfileV2
from core.planning.planner import build_plan_from_capability_map
from core.planning.verifier import verify_plan
from core.planning.renderer import render_plan_for_user
from core.responses import make_assistant_response, make_response_update
from core.planning.execution_queue import (
    find_next_executable_step,
    mark_plan_step_started,
    mark_plan_step_after_execution,
)
from core.deliverables.gate import evaluate_deliverable_gate_state
from core.deliverables.evidence import extract_final_answer_content_from_state
from core.audit.execution_state import audit_execution_state

def _as_plain_dict(obj):
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    return {}


def _extract_data_version_update(raw_execution):
    """
    Extract plugin data_version_update from ToolExecutionResult.

    New protocol:
    execute_analysis_tool() places plugin-level data_version_update into
    ToolExecutionResult.payload["data_version_update"].

    We also check top-level for defensive compatibility, but the canonical
    path is payload.data_version_update.
    """
    raw = _as_plain_dict(raw_execution)

    update = raw.get("data_version_update")
    if update is not None:
        return update

    payload = raw.get("payload", {}) or {}
    payload = _as_plain_dict(payload)

    return payload.get("data_version_update")


def _validate_data_version_update(data_version_update):
    if not data_version_update:
        return None

    if not isinstance(data_version_update, dict):
        return None

    new_version = data_version_update.get("new_version")
    active_data_version_id = data_version_update.get("active_data_version_id")

    if not isinstance(new_version, dict):
        return None

    new_version_id = (
        data_version_update.get("new_version_id")
        or new_version.get("version_id")
    )

    if not new_version_id:
        return None

    if not active_data_version_id:
        return None

    if active_data_version_id != new_version_id:
        return None

    if not new_version.get("version_id"):
        new_version["version_id"] = new_version_id

    return {
        **data_version_update,
        "new_version_id": new_version_id,
        "active_data_version_id": active_data_version_id,
        "new_version": new_version,
    }

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

def _merge_state_for_audit(state: GraphState, updates: dict) -> dict:
    """
    Build a best-effort post-node state snapshot for backend audit.

    This does not mutate graph state. It only gives audit_execution_state()
    a view of what state will look like after this node's updates.
    """
    merged = dict(state)

    for key, value in updates.items():
        if key in {"observations", "analysis_runs", "data_versions", "data_audit_log"}:
            old_value = merged.get(key, []) or []
            new_value = value or []

            if isinstance(old_value, list) and isinstance(new_value, list):
                merged[key] = old_value + new_value
            else:
                merged[key] = value
        else:
            merged[key] = value

    return merged


def _attach_execution_audit(state: GraphState, updates: dict) -> dict:
    """
    Run backend execution-state audit after a node update.

    S11B is observe-only: audit findings are recorded but do not alter routing.
    """
    audit_state = _merge_state_for_audit(state, updates)
    audit_result = audit_execution_state(audit_state)

    updates["execution_audit"] = audit_result.model_dump()

    if audit_result.status != "ok":
        print("\n" + "=" * 40)
        print("[EXECUTION AUDIT]")
        print(audit_result.model_dump())
        print("=" * 40 + "\n")

    return updates

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

def intent_router_node(state: GraphState):
    user_request = state.get("user_request", "")
    intent = classify_interaction_intent(user_request)

    print("\n" + "=" * 40)
    print("[INTENT ROUTER]")
    print(f"user_request = {user_request}")
    print(f"intent = {intent.value}")
    print("=" * 40 + "\n")

    return {
        "interaction_intent": intent.value,
    }

def advisory_answer_node(state: GraphState):
    summary = state.get("dataset_summary") or {}
    capability_map = state.get("capability_map") or {}

    n_rows = summary.get("n_rows", "unknown")
    n_cols = summary.get("n_cols", "unknown")

    numeric_cols = summary.get("numeric_columns", []) or []
    categorical_cols = summary.get("categorical_columns", []) or []
    binary_cols = summary.get("binary_columns", []) or []
    id_like_cols = summary.get("id_like_columns", []) or []
    missingness = summary.get("missingness_summary", {}) or {}

    capabilities = capability_map.get("capabilities", []) or []

    ready = [c for c in capabilities if c.get("status") == "ready"]
    needs_choice = [c for c in capabilities if c.get("status") == "needs_user_choice"]
    not_applicable = [
        c for c in capabilities
        if c.get("status") in {"not_applicable", "blocked"}
    ]

    lines = []

    lines.append("I have profiled the current dataset. Here is what you can do next.")
    lines.append("")
    lines.append("Dataset overview:")
    lines.append(f"- Rows: {n_rows}")
    lines.append(f"- Columns: {n_cols}")
    lines.append(f"- Numeric columns: {len(numeric_cols)}")
    lines.append(f"- Categorical columns: {len(categorical_cols)}")
    lines.append(f"- Binary columns: {len(binary_cols)}")
    lines.append(f"- ID-like columns: {len(id_like_cols)}")
    lines.append(
        f"- Columns with missing values: "
        f"{missingness.get('n_columns_with_missing', 0)}"
    )
    lines.append("")

    if ready:
        lines.append("Analyses that appear ready:")
        for cap in ready[:8]:
            lines.append(f"- {cap.get('display_name', cap.get('tool_name'))}: {cap.get('reason')}")
        lines.append("")

    if needs_choice:
        lines.append("Analyses that may be useful but need your choices first:")
        for cap in needs_choice[:8]:
            choices = ", ".join(cap.get("required_user_choices", []) or [])
            lines.append(
                f"- {cap.get('display_name', cap.get('tool_name'))}: "
                f"needs {choices or 'additional choices'}"
            )
        lines.append("")

    if not_applicable:
        lines.append("Currently blocked or not recommended:")
        for cap in not_applicable[:5]:
            lines.append(f"- {cap.get('display_name', cap.get('tool_name'))}: {cap.get('reason')}")
        lines.append("")

    lines.append("I have not run any analysis tools yet.")
    lines.append("If you want, say `make a plan` and I will draft a data-aware plan without executing it.")

    answer = "\n".join(lines)

    updates = make_response_update(
        response_type="advisory",
        content=answer,
        source_node="advisory_answer",
        data_version_id=state.get("active_data_version_id"),
        metadata={
            "interaction_intent": state.get("interaction_intent"),
        },
    )

    updates.update({
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
    })

    return updates

def plan_only_node(state: GraphState):
    capability_map_dict = state.get("capability_map")
    profile_dict = state.get("dataset_profile_v2")

    if not capability_map_dict or not profile_dict:
        content = (
            "I cannot create a data-aware plan yet because the dataset profile "
            "is not available. Please upload or reload a dataset first."
        )

        updates = make_response_update(
            response_type="error",
            content=content,
            source_node="plan_only",
            data_version_id=state.get("active_data_version_id"),
            metadata={
                "reason": "missing_dataset_profile_or_capability_map",
            },
        )

        updates.update({
            "pending_plan": None,
            "plan_status": "blocked",
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
        })

        return updates

    capability_map = CapabilityMap.model_validate(capability_map_dict)
    dataset_profile = DatasetProfileV2.model_validate(profile_dict)

    plan = build_plan_from_capability_map(
        user_request=state.get("user_request", ""),
        capability_map=capability_map,
    )

    verified_plan = verify_plan(plan, dataset_profile)
    rendered = render_plan_for_user(verified_plan)

    print("\n" + "=" * 40)
    print("[PLAN ONLY NODE]")
    print(f"plan_id = {verified_plan.plan_id}")
    print(f"plan_status = {verified_plan.status}")
    print(f"n_steps = {len(verified_plan.steps)}")
    print(f"n_blocked = {len(verified_plan.blocked_or_not_recommended)}")
    print("=" * 40 + "\n")

    updates = make_response_update(
        response_type="plan",
        content=rendered,
        source_node="plan_only",
        data_version_id=state.get("active_data_version_id"),
        plan_id=verified_plan.plan_id,
        plan_status=verified_plan.status,
        metadata={
            "interaction_intent": state.get("interaction_intent"),
            "n_steps": len(verified_plan.steps),
            "n_blocked": len(verified_plan.blocked_or_not_recommended),
        },
    )

    updates.update({
        "pending_plan": verified_plan.model_dump(),
        "plan_status": verified_plan.status,

        # Hard safety reset.
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,
    })

    return updates

def route_after_intent(state: GraphState):
    intent = state.get("interaction_intent")

    print(f"[ROUTE AFTER INTENT] intent = {intent}")

    if intent == "advisory":
        return "advisory_answer"

    if intent == "plan_only":
        return "plan_only"

    if intent == "execute_plan":
        return "execute_pending_plan"

    # Direct tool requests and unknown requests go to the unified supervisor path.
    return "supervisor"

def execute_pending_plan_node(state: GraphState):
    print("\n" + "=" * 40)
    print("[EXECUTE PENDING PLAN NODE ENTERED]")
    print(f"plan_status = {state.get('plan_status')}")
    print(f"has_pending_plan = {state.get('pending_plan') is not None}")
    print("=" * 40 + "\n")

    pending_plan = state.get("pending_plan")

    if not pending_plan:
        content = (
            "There is no pending plan to execute. "
            "Please ask me to make a plan first."
        )

        updates = make_response_update(
            response_type="plan_execution_status",
            content=content,
            source_node="execute_pending_plan",
            data_version_id=state.get("active_data_version_id"),
            metadata={
                "reason": "no_pending_plan",
            },
        )

        updates.update({
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
            "plan_execution_status": "no_pending_plan",
        })

        return updates

    next_step, readiness = find_next_executable_step(
        pending_plan,
        profile=state.get("dataset_profile"),
    )

    if next_step is None or readiness is None or not readiness.executable:
        reason = readiness.reason if readiness is not None else "No candidate step found."
        missing_choices = readiness.missing_user_choices if readiness is not None else []

        lines = []
        lines.append("The pending plan has no execution-ready steps.")
        lines.append("")
        lines.append(f"Reason: {reason}")

        if missing_choices:
            lines.append("")
            lines.append("Missing choices:")
            for choice in missing_choices:
                lines.append(f"- {choice}")

        lines.append("")
        lines.append("No tools were executed.")

        content = "\n".join(lines)

        updates = make_response_update(
            response_type="plan_execution_status",
            content=content,
            source_node="execute_pending_plan",
            data_version_id=state.get("active_data_version_id"),
            plan_id=pending_plan.get("plan_id"),
            plan_status=pending_plan.get("status"),
            metadata={
                "reason": "no_executable_step",
                "readiness": readiness.model_dump() if readiness is not None else None,
            },
        )

        updates.update({
            "plan_execution_status": "blocked_no_ready_steps",
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
        })

        return updates

    action = readiness.action

    updated_plan = mark_plan_step_started(
        pending_plan,
        step_id=next_step["step_id"],
        action_id=action.action_id,
    )

    print("\n" + "=" * 40)
    print("[EXECUTE PENDING PLAN]")
    print(f"plan_id = {pending_plan.get('plan_id')}")
    print(f"step_id = {next_step.get('step_id')}")
    print(f"tool_name = {action.tool_name}")
    print(f"arguments = {action.arguments}")
    print(f"readiness_status = {readiness.status}")
    print("=" * 40 + "\n")

    return {
        "pending_plan": updated_plan,
        "plan_status": updated_plan.get("status"),
        "current_plan_step_id": next_step["step_id"],
        "plan_execution_status": "started_step",

        # Important for S4:
        # This action came from a pending plan, not a direct user tool request.
        "action_origin": "pending_plan",

        # Existing verify -> human_review / execute path.
        "current_action": action,
        "current_execution": None,
        "current_verification": None,
    }

def route_after_execute_pending_plan(state: GraphState):
    action = state.get("current_action")

    if action is not None:
        return "verify"

    return "end"


def supervisor_node(state: GraphState):
    current_workspace = state.get("workspace_dir", "./")
    current_profile = state.get("dataset_profile")

    context_pkg = build_context(
        step=state.get("current_step", 1),
        max_steps=state.get("max_steps", 12),
        user_request=state.get("user_request", "Not provided"),
        profile=current_profile,
        observations=state.get("observations", []),
        workspace_dir=current_workspace,
        deliverable_check=state.get("deliverable_check"),
        data_versions=state.get("data_versions", []),
        active_data_version_id=state.get("active_data_version_id"),
        data_audit_log=state.get("data_audit_log", []),
    )

    action = call_supervisor(context_pkg)
    updates = {"current_action": action}

    print("\n" + "=" * 40)
    print(f"[Supervisor decision]: action_type = {action.action_type}")
    print(f"[Reasoning summary]: {action.reasoning_summary}")
    print("=" * 40 + "\n")

    contract = getattr(action, "task_contract", None)
    if contract is not None:
        if hasattr(contract, "model_dump"):
            contract_dict = contract.model_dump()
        elif isinstance(contract, dict):
            contract_dict = contract
        else:
            contract_dict = {}

        print(
            f"[TASK CONTRACT DECLARED] "
            f"deliverables={len(contract_dict.get('required_deliverables', []))}"
        )

        updates["task_contract"] = contract_dict

    return updates




def verify_node(state: GraphState):
    """
    Verification node.

    New contract:
    verify() returns a full VerificationResult produced from
    analysis_tool_plugins.validation.
    """
    action = state["current_action"]

    status, feedback, verify_result = verify(action, state["dataset_profile"])

    action_hash = get_action_hash(
        getattr(action, "tool_name", None),
        getattr(action, "arguments", {}) or {},
    )

    verify_result.details["action_hash"] = action_hash

    print("\n" + "=" * 40)
    print("[VERIFY NODE DEBUG]")
    print(f"tool_name = {getattr(action, 'tool_name', None)}")
    print(f"verify_result.status = {verify_result.status}")
    print(f"verify_result.error_code = {verify_result.error_code}")
    print(f"verify_result.feedback = {verify_result.feedback}")
    print(f"verify_result.details = {verify_result.details}")
    print("=" * 40 + "\n")

    if verify_result.status in ["rejected_recoverable", "rejected_terminal"]:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action.action_id,
            tool_name=getattr(action, "tool_name", None),
            arguments=getattr(action, "arguments", {}) or {},
            status="rejected",
            success=False,
            error_code=verify_result.error_code or "VERIFICATION_FAILED",
            message=verify_result.feedback,
            artifacts=[],
            summary=(
                f"Validation failed for {getattr(action, 'tool_name', None)}: "
                f"{verify_result.feedback}"
            ),
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": verify_result.error_code or "VERIFICATION_FAILED",
                "message": verify_result.feedback,
                "details": verify_result.details,
            },
            raw_data={
                "verification": (
                    verify_result.model_dump()
                    if hasattr(verify_result, "model_dump")
                    else verify_result
                ),
                "recoverable": verify_result.status == "rejected_recoverable",
            },
        )

        updates = {
            "current_verification": verify_result,
            "observations": [obs.model_dump()],
        }

        # Phase 4 safety:
        # If a pending-plan step fails verification, mark that step failed and stop.
        # Do not loop back into build_context with the same "run the plan" request.
        current_plan_step_id = state.get("current_plan_step_id")
        pending_plan = state.get("pending_plan")

        if current_plan_step_id and pending_plan:
            updated_plan = mark_plan_step_after_execution(
                pending_plan,
                step_id=current_plan_step_id,
                success=False,
                execution_id=None,
                message=verify_result.feedback,
            )

            content = (
                "I tried to execute the next ready step in the pending plan, "
                "but it failed validation before execution.\n\n"
                f"Tool: {getattr(action, 'tool_name', None)}\n"
                f"Reason: {verify_result.feedback}\n\n"
                "I marked this plan step as failed and stopped execution to avoid a retry loop."
            )

            updates.update({
                "pending_plan": updated_plan,
                "plan_status": updated_plan.get("status"),
                "current_plan_step_id": None,
                "current_action": None,
                "current_execution": None,
                "plan_execution_status": "step_verification_failed",
                "assistant_response": make_assistant_response(
                    response_type="error",
                    content=content,
                    source_node="verify",
                    data_version_id=state.get("active_data_version_id"),
                    plan_id=updated_plan.get("plan_id"),
                    plan_status=updated_plan.get("status"),
                    metadata={
                        "error_code": verify_result.error_code,
                        "tool_name": getattr(action, "tool_name", None),
                        "step_id": current_plan_step_id,
                    },
                ),
            })

        return updates

    return {
        "current_verification": verify_result,
    }


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

    tool_name = getattr(action, "tool_name", None)
    arguments = getattr(action, "arguments", {}) or {}

    if isinstance(vr, dict):
        vr_details = vr.get("details", {}) or {}
    else:
        vr_details = getattr(vr, "details", {}) or {}

    canonical_arguments = vr_details.get("canonical_arguments") or arguments
    vr_status = getattr(vr, "status", None)
    feedback = getattr(vr, "feedback", None)

    # Case 0: user approved the pending action.
    # Because the graph was interrupted before human_review,
    # after approval it resumes here first. We do not create an observation here.
    # Routing after human_review will send it to execute.
    if vr_status == "allowed":
        print("[HUMAN REVIEW] User approved action; routing to execute.")
        return {}

    # Case 1: high-risk tool needs user confirmation
    if vr_status == "needs_review":
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action.action_id,
            tool_name=tool_name,
            arguments=arguments,
            status="rejected",
            success=False,
            error_code="HUMAN_CONFIRMATION_REQUIRED",
            message=feedback or f"Tool {tool_name} requires human confirmation.",
            artifacts=[],
            summary=(
                f"Tool {tool_name} requires human confirmation and was not executed. "
                f"Arguments: {canonical_arguments}. Feedback: {feedback}"
            ),
            structured_data={
                "status": "needs_review",
                "success": False,
                "error_code": "HUMAN_CONFIRMATION_REQUIRED",
                "message": feedback,
                "pending_action": action.model_dump() if hasattr(action, "model_dump") else {},
            },
            raw_data={
                "verification": vr.model_dump() if hasattr(vr, "model_dump") else {},
                "pending_action": action.model_dump() if hasattr(action, "model_dump") else {},
            },
        )

        return {
            "human_review_required": True,
            "pending_action": action.model_dump() if hasattr(action, "model_dump") else action,
            "observations": [obs.model_dump()],
        }

    # Case 2: rejected by verifier
    if vr_status in {"rejected_recoverable", "rejected_terminal"}:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action.action_id,
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
                "verification": vr.model_dump() if hasattr(vr, "model_dump") else {},
            },
        )

        return {"observations": [obs.model_dump()]}


    # Safety fallback
    obs = Observation(
        observation_id=f"obs_{uuid.uuid4().hex[:8]}",
        source_action_id=action.action_id,
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
            "verification": vr.model_dump() if hasattr(vr, "model_dump") else {},
        },
    )

    return {"observations": [obs.model_dump()]}


def execute_node(state: GraphState):
    action = state.get("current_action")

    if not action or not hasattr(action, "tool_name"):
        return {"current_execution": "Error: No valid action provided."}

    # 1. Current action fingerprint
    current_hash = get_action_hash(action.tool_name, action.arguments)

    # 2. Fingerprints from prior observations
    executed_hashes = []
    for obs in state.get("observations", []):
        if isinstance(obs, dict) and obs.get('tool_name'):
            # Legacy observations without arguments default to {}
            obs_args = obs.get('arguments', {})
            executed_hashes.append(get_action_hash(obs['tool_name'], obs_args))

    # 3. Fingerprint gate: block identical parameters
    if current_hash in executed_hashes:
        error_msg = (
            f"[System intervention]: 🚫 Execution refused. You are calling '{action.tool_name}' "
            f"with parameters identical to a previous attempt.\n"
            f"To retry, change arguments (e.g. add .dropna() in chart code or change cleaning strategy)."
        )
        print(f"[Fingerprint gate]: blocked duplicate {action.tool_name} (fp: {current_hash[:6]})")
        return {"current_execution": error_msg}

    print(f"[Execute]: {action.tool_name}")
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

    if hasattr(exec_result, 'model_dump'):
        raw_payload = exec_result.model_dump()
    elif hasattr(exec_result, 'dict'):
        raw_payload = exec_result.dict()
    else:
        raw_payload = exec_result

    safe_result = sanitize_results(raw_payload)

    return {"current_execution": safe_result}

def summarize_node(state: GraphState):
    current_action = state.get("current_action")
    tool_name = current_action.tool_name if current_action else "unknown_tool"

    arguments = {}
    if current_action and hasattr(current_action, "arguments"):
        arguments = current_action.arguments

    raw_result = state.get("current_execution", "No execution result")

    if isinstance(raw_result, dict):
        execution_id = raw_result.get("execution_id")
        status = raw_result.get("status", "ok" if raw_result.get("success", True) else "failed")
        success = bool(raw_result.get("success", status in ["ok", "warning"]))
        error_code = raw_result.get("error_code")
        message = raw_result.get("message")
        artifacts = raw_result.get("artifacts", []) or []
        payload = raw_result.get("payload", {})
    else:
        execution_id = None
        status = "failed"
        success = False
        error_code = "NON_STRUCTURED_EXECUTION_RESULT"
        message = str(raw_result)
        artifacts = []
        payload = {"result": raw_result}

    summary = (
        f"Tool {tool_name} finished with status={status}, success={success}. "
        f"message={message or 'No message'}"
    )

    if error_code:
        summary += f" error_code={error_code}."

    refined_observation = {
        "observation_id": f"obs_{uuid.uuid4().hex[:8]}",
        "source_action_id": getattr(current_action, "action_id", "unknown"),
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
    data_version_update = _extract_data_version_update(raw_result)
    validated_version_update = _validate_data_version_update(data_version_update)

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

        existing_runs = state.get("analysis_runs", []) or []
        updates["analysis_runs"] = existing_runs + [analysis_run]


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

    return _attach_execution_audit(state, updates)


def final_response_node(state: GraphState):
    """
    Convert a deliverable-gate-approved final answer into assistant_response.

    DeliverableGate remains the quality gate.
    This node is only the output-envelope adapter.
    """
    content = extract_final_answer_content_from_state(state)

    if not content:
        content = (
            "The final answer passed the deliverable gate, but no final-answer "
            "content could be extracted from the current graph state."
        )

        updates = make_response_update(
            response_type="error",
            content=content,
            source_node="final_response",
            data_version_id=state.get("active_data_version_id"),
            metadata={
                "reason": "missing_final_answer_content",
                "deliverable_check": state.get("deliverable_check"),
            },
        )

        updates.update({
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
        })

        return updates

    updates = make_response_update(
        response_type="final_answer",
        content=content,
        source_node="final_response",
        data_version_id=state.get("active_data_version_id"),
        metadata={
            "deliverable_check": state.get("deliverable_check"),
        },
    )

    updates.update({
        # Clear completed final-answer action.
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
    })

    return updates

# --- Routing ---
def route_after_supervisor(state: GraphState):
    """
    After supervisor:
    - tool_call -> verify
    - final_answer / ask_user -> deliverable_gate
    - max_steps reached -> end
    """
    action = state.get("current_action")

    if action and hasattr(action, "action_type") and action.action_type in ["final_answer", "ask_user"]:
        print("[ROUTE AFTER SUPERVISOR] final_answer -> deliverable_gate")
        return "deliverable_gate"

    if state.get("current_step", 0) >= state.get("max_steps", 12):
        print("[ROUTE AFTER SUPERVISOR] max_steps -> end")
        return "end"

    return "verify"

def route_after_verify(state: GraphState):
    """
    After verification:
    - allowed: execute the tool
    - needs_review: interrupt before human_review and wait for user approval
    - rejected_*: do not execute; go back to build_context so Supervisor can rethink/respond
    """

    # S4: if a pending-plan action fails verification,
    # do not loop back and continue the same "run the plan" turn.
    if (
            state.get("action_origin") == "pending_plan"
            and state.get("current_verification") is not None
    ):
        verification = state.get("current_verification")
        status = getattr(verification, "status", None)

        if status in {"rejected_recoverable", "rejected_terminal"}:
            return "end"

    if state.get("plan_execution_status") == "step_verification_failed":
        return "end"

    vr = state.get("current_verification")

    if vr is None:
        print("[ROUTE AFTER VERIFY] no verification result -> build_context")
        return "build_context"

    if isinstance(vr, dict):
        status = vr.get("status")
    else:
        status = getattr(vr, "status", None)

    print(f"[ROUTE AFTER VERIFY] status = {status}")

    if status == "allowed":
        return "execute"

    if status == "needs_review":
        return "human_review"

    if status in {"rejected_recoverable", "rejected_terminal"}:
        return "build_context"

    return "build_context"


def route_after_review(state: GraphState):
    """
    After human_review:
    - if user approved, execute the original pending action
    - otherwise go back to build_context and let Supervisor rethink/respond
    """
    vr = state.get("current_verification")

    if vr is None:
        print("[ROUTE AFTER REVIEW] no current_verification -> build_context")
        return "build_context"

    if isinstance(vr, dict):
        status = vr.get("status")
    else:
        status = getattr(vr, "status", None)

    print(f"[ROUTE AFTER REVIEW] status = {status}")

    if status == "allowed":
        return "execute"

    return "build_context"


def route_after_summarize(state: GraphState):
    # S4: a single "run the plan" turn executes at most one PlanStep.
    # If the action came from pending_plan, stop after summarize.
    if state.get("action_origin") == "pending_plan":
        return "end"

    if state.get("current_step", 0) >= state.get("max_steps", 12):
        return "end"

    observations = state.get("observations", []) or []
    last_obs = observations[-1] if observations else {}

    status = last_obs.get("status") if isinstance(last_obs, dict) else None
    error_code = last_obs.get("error_code") if isinstance(last_obs, dict) else None

    raw_data = last_obs.get("raw_data", {}) if isinstance(last_obs, dict) else {}
    recoverable = False

    if isinstance(raw_data, dict):
        recoverable = bool(raw_data.get("recoverable", False))

    # Successful or warning result: continue normal loop.
    if status in {"ok", "warning"}:
        return "build_context"

    # Human confirmation required is an interrupt/review state, not a tool failure.
    if error_code == "HUMAN_CONFIRMATION_REQUIRED":
        return "end"

    # Recoverable tool/schema failures can go back once or twice.
    # For now, use max_steps as the safety brake.
    if status in {"blocked", "failed", "rejected"} and recoverable:
        return "build_context"

    # Non-recoverable failures should stop and let final answer/report explain blocker.
    return "end"

def sanitize_results(obj):
    """
    Recursively convert numpy scalars/arrays to native Python for serialization.
    """
    if isinstance(obj, dict):
        return {k: sanitize_results(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_results(v) for v in obj]
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Pass through primitives
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

def get_action_hash(tool_name: str, arguments: dict):
    """Stable MD5 fingerprint from tool name + canonical JSON arguments."""
    if not arguments:
        arguments = {}
    # sort_keys keeps fingerprint stable when key order changes
    arg_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(f"{tool_name}_{arg_str}".encode('utf-8')).hexdigest()

def deliverable_gate_node(state: GraphState):
    result = evaluate_deliverable_gate_state(state)

    deliverable_check = result.model_dump()

    print("\n" + "=" * 40)
    print("[DELIVERABLE GATE]")
    print(deliverable_check)
    print("=" * 40 + "\n")

    return {
        "deliverable_check": deliverable_check,
    }

def route_after_deliverable_gate(state: GraphState):
    """
    If deliverables are satisfied, allow final_answer to end.
    If deliverables are missing, go back to build_context so Supervisor can continue.
    """
    deliverable_check = state.get("deliverable_check") or {}

    if isinstance(deliverable_check, dict):
        status = deliverable_check.get("status")
    else:
        status = getattr(deliverable_check, "status", None)

    print(f"[ROUTE AFTER DELIVERABLE GATE] status = {status}")

    if status == "ok":
        return "final_response"

    if status in {"needs_more_work", "missing", "blocked"}:
        return "build_context"

    return "end"


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



# Compile with checkpoint + interrupt
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])