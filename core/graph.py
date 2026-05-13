import os
from verifiers.validators import verify
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import re
import numpy as np

from core.state import GraphState
# Import VerificationResult for verify_node.
from core.schema import Observation, ContextPackage, VerificationResult
# Import generate_profile for build_context_node.
from core.context_builder import build_context, generate_profile
from agents.supervisor import call_supervisor
from core.analysis_tool_plugins.execution import execute_tool
import hashlib
import json
from core.deliverables import check_deliverables, check_answer_quality
from core.analysis_runs import build_analysis_run_from_observation

# --- Graph nodes ---
def build_context_node(state: GraphState):
    step = state.get("current_step", 0) + 1

    current_workspace = state["workspace_dir"]
    # Resolve data file dynamically (any working_data* in sandbox).
    current_data_path = None
    for file in os.listdir(current_workspace):
        if file.startswith("working_data"):
            current_data_path = os.path.join(current_workspace, file)
            break

    if not current_data_path:
        # No CSV/DataFrame dataset is loaded.
        # This is allowed because the agent may still answer general questions
        # or use SQL tools such as inspect_sql_schema / run_sql_query.
        user_request = state.get("user_request", "")

        observations = state.get("observations", []) or []
        recent_history = ""

        for obs in observations[-10:]:
            if isinstance(obs, dict):
                tool_name = obs.get("tool_name", "unknown_tool")
                status = obs.get("status", "unknown")
                summary = obs.get("summary") or obs.get("message") or ""
                recent_history += f"- {tool_name} [{status}]: {summary}\n"

        if not recent_history:
            recent_history = "(No previous observations.)"

        context_text = (
            f"User request:\n{user_request}\n\n"
            "Current data context:\n"
            "- No in-memory CSV/DataFrame dataset is currently loaded.\n"
            "- DataFrame-specific tools such as summary statistics, missingness_report, "
            "regression, scatterplot, and residual plots require an uploaded dataset.\n"
            "- The user may still ask general questions or ask to inspect/analyze a SQL database.\n"
            "- If the user provides a DuckDB database path, prefer SQL tools such as "
            "`inspect_sql_schema` and `run_sql_query`.\n\n"
            f"Recent observations:\n{recent_history}\n"
        )

        return {
            "current_context_text": context_text,
            "dataset_profile": None,

            # Keep existing state fields stable.
            "workspace_dir": current_workspace,
            "data_versions": state.get("data_versions", []),
            "active_data_version_id": state.get("active_data_version_id"),
            "data_audit_log": state.get("data_audit_log", []),
            "analysis_runs": state.get("analysis_runs", []),

            # Old hard-gating state should not be revived.
            "task_contract": None,
            "deliverable_check": None,
            "deliverable_gate_attempts": 0,
        }

    # Refresh dataset profile from sandbox.
    new_profile = generate_profile(current_data_path)

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
        "dataset_profile": new_profile
    }


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

    # Stabilization mode:
    # task_contract is disabled. The Supervisor should operate one action at a time.
    # Do not persist task_contract into graph state.

    # contract = getattr(action, "task_contract", None)
    # if contract is not None:
    #     if hasattr(contract, "model_dump"):
    #         contract_dict = contract.model_dump()
    #     elif isinstance(contract, dict):
    #         contract_dict = contract
    #     else:
    #         contract_dict = {}
    #
    #     print(
    #         f"[TASK CONTRACT DECLARED] "
    #         f"deliverables={len(contract_dict.get('required_deliverables', []))}"
    #     )
    #
    #     updates["task_contract"] = contract_dict

    return updates




def verify_node(state: GraphState):
    """
    Verification node: run verify().

    Always persist current_verification for allowed / needs_review / rejected
    so routing does not mis-send low-risk tools to human_review.
    """
    action = state["current_action"]

    profile = state.get("dataset_profile")
    status, feedback = verify(action, profile=profile, state=state)

    verify_result = VerificationResult(
        action_id=action.action_id,
        status=status,
        feedback=feedback
    )

    print("\n" + "=" * 40)
    print("[VERIFY NODE DEBUG]")
    print(f"tool_name = {getattr(action, 'tool_name', None)}")
    print(f"verify_result.status = {verify_result.status}")
    print(f"verify_result.feedback = {verify_result.feedback}")
    print("=" * 40 + "\n")

    if verify_result.status in ["rejected_recoverable", "rejected_terminal"]:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action.action_id,
            tool_name=getattr(action, "tool_name", None),
            arguments=getattr(action, "arguments", {}) or {},
            status="rejected",
            success=False,
            error_code=getattr(verify_result, "error_code", None) or "VERIFICATION_FAILED",
            message=verify_result.feedback,
            artifacts=[],
            summary=f"Validation failed for {getattr(action, 'tool_name', None)}: {verify_result.feedback}",
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": getattr(verify_result, "error_code", None) or "VERIFICATION_FAILED",
                "message": verify_result.feedback,
            },
            raw_data={"verification": verify_result.model_dump()},
        )

        return {
            "current_verification": verify_result,
            "observations": [obs.model_dump()]
        }

    # allowed and needs_review must also return current_verification
    return {
        "current_verification": verify_result
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
                f"Arguments: {arguments}. Feedback: {feedback}"
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
        return {
            "current_execution": {
                "execution_id": f"exec_{uuid.uuid4().hex[:8]}",
                "action_id": "unknown",
                "tool_name": None,
                "status": "failed",
                "success": False,
                "error_code": "NO_VALID_ACTION",
                "message": "No valid action provided to execute_node.",
                "recoverable": True,
                "data_version_id": state.get("active_data_version_id"),
                "data_version_update": None,
                "payload": {},
                "artifacts": [],
                "raw_payload": {},
            }
        }

    # 1. Current action fingerprint
    current_hash = get_action_hash(action.tool_name, action.arguments)

    # 2. Fingerprints from prior successful observations on the same active data version.
    # Do not block retries after failed/rejected runs.
    # Do not block reruns after data version changed.
    active_version_id = state.get("active_data_version_id")
    executed_hashes = []

    for obs in state.get("observations", []):
        if not isinstance(obs, dict) or not obs.get("tool_name"):
            continue

        obs_status = obs.get("status")
        obs_success = obs.get("success")
        obs_version = obs.get("data_version_id")

        structured = obs.get("structured_data", {}) or {}
        if obs_version is None and isinstance(structured, dict):
            obs_version = structured.get("data_version_id")

        if obs_status not in {"ok", "warning"} and obs_success is not True:
            continue

        if active_version_id and obs_version and obs_version != active_version_id:
            continue

        obs_args = obs.get("arguments", {}) or {}
        executed_hashes.append(get_action_hash(obs["tool_name"], obs_args))

    # 3. Fingerprint gate: block identical successful tool calls only.
    if current_hash in executed_hashes:
        error_msg = (
            f"Duplicate tool call blocked: `{action.tool_name}` was already executed "
            f"successfully with the same arguments on the current data version. "
            f"Choose a different tool, different arguments, or provide a final answer from the existing result."
        )

        print(f"[Fingerprint gate]: blocked duplicate {action.tool_name} (fp: {current_hash[:6]})")

        return {
            "current_execution": {
                "status": "blocked",
                "success": False,
                "error_code": "DUPLICATE_SUCCESSFUL_TOOL_CALL",
                "message": error_msg,
                "recoverable": True,
                "payload": {
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                    "active_data_version_id": active_version_id,
                },
                "artifacts": [],
            }
        }

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

    exec_result = execute_tool(action, context_pkg)

    if hasattr(exec_result, 'model_dump'):
        raw_payload = exec_result.model_dump()
    elif hasattr(exec_result, 'dict'):
        raw_payload = exec_result.dict()
    else:
        raw_payload = exec_result

    safe_result = sanitize_results(raw_payload)

    if hasattr(safe_result, "model_dump"):
        safe_result = safe_result.model_dump()

    return {"current_execution": safe_result}

def summarize_node(state: GraphState):
    current_action = state.get("current_action")
    tool_name = current_action.tool_name if current_action else "unknown_tool"

    arguments = {}
    if current_action and hasattr(current_action, "arguments"):
        arguments = current_action.arguments

    raw_result = state.get("current_execution", "No execution result")

    if isinstance(raw_result, dict):
        status = raw_result.get("status", "ok" if raw_result.get("success", True) else "failed")
        success = bool(raw_result.get("success", status in {"ok", "warning"}))
        error_code = raw_result.get("error_code")
        message = raw_result.get("message")
        recoverable = bool(raw_result.get("recoverable", False))
        artifacts = raw_result.get("artifacts", []) or []
        payload = raw_result.get("payload", {}) or {}
        data_version_id = raw_result.get("data_version_id") or state.get("active_data_version_id")
        data_version_update = raw_result.get("data_version_update")

        if data_version_update is None and isinstance(payload, dict):
            data_version_update = payload.get("data_version_update")

    else:
        status = "failed"
        success = False
        error_code = "NON_STRUCTURED_EXECUTION_RESULT"
        message = str(raw_result)
        recoverable = True
        artifacts = []
        payload = {"result": raw_result}
        data_version_id = state.get("active_data_version_id")
        data_version_update = None

    summary = (
        f"Tool {tool_name} finished with status={status}, success={success}. "
        f"message={message or 'No message'}"
    )

    if error_code:
        summary += f" error_code={error_code}."

    refined_observation_model = Observation(
        observation_id=f"obs_{uuid.uuid4().hex[:8]}",
        source_action_id=getattr(current_action, "action_id", "unknown"),
        tool_name=tool_name,
        arguments=arguments,

        data_version_id=data_version_id,

        status=status,
        success=success,
        error_code=error_code,
        message=message,
        recoverable=recoverable,
        artifacts=artifacts,
        summary=summary,
        structured_data={
            "status": status,
            "success": success,
            "error_code": error_code,
            "message": message,
            "recoverable": recoverable,
            "artifacts": artifacts,
            "payload": payload,
            "data_version_id": data_version_id,
            "data_version_update": data_version_update,
        },
        raw_data=raw_result if isinstance(raw_result, dict) else {"result": raw_result},
    )

    refined_observation = refined_observation_model.model_dump()

    print(f"[Summarize]: archived result for {tool_name}.")

    updates = {
        "observations": [refined_observation],

        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,

        "current_step": state.get("current_step", 0) + 1,

    }

    # Phase 3: append successful tool result to Analysis Results registry
    if status in {"ok", "warning", "blocked"} and tool_name not in {"unknown_tool"}:
        analysis_run = build_analysis_run_from_observation(
            tool_name=tool_name,
            action_id=getattr(current_action, "action_id", "unknown"),
            arguments=arguments,
            data_version_id=data_version_id,
            status=status,
            success=success,
            message=message,
            payload=payload,
            artifacts=artifacts,
            observation_id=refined_observation["observation_id"],
        )

        existing_runs = state.get("analysis_runs", []) or []
        updates["analysis_runs"] = existing_runs + [analysis_run]

    if data_version_update:
        new_version = data_version_update.get("new_version")
        new_active_id = data_version_update.get("active_data_version_id")
        audit_event = data_version_update.get("audit_event")

        existing_versions = state.get("data_versions", []) or []
        existing_audit_log = state.get("data_audit_log", []) or []

        if new_version:
            updates["data_versions"] = existing_versions + [new_version]

        if new_active_id:
            updates["active_data_version_id"] = new_active_id

        if audit_event:
            updates["data_audit_log"] = existing_audit_log + [audit_event]

        print(f"[DATA VERSION] active_data_version_id -> {new_active_id}")

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
        print("[ROUTE AFTER SUPERVISOR] final_answer -> answer_quality_gate")
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
    if state.get("current_step", 0) >= state.get("max_steps", 12):
        return "end"
    return "build_context"

def call_llm_to_route(state: GraphState):
    """
    Semantic routing stub (replace with LLM call).
    """
    prompt = f"""
    Classify the user's task intent.

    User request: "{state['user_request']}"
    Number of columns: {len(state.get('dataset_profile').columns) if state.get('dataset_profile') else 0}

    Rules:
    - Simple lookups (single values, column names, row counts, univariate stats) -> reply 'SUPERVISOR'.
    - Multivariate analysis, modeling, plotting, prediction, exploration -> reply 'PLANNER'.

    Reply only 'PLANNER' or 'SUPERVISOR'.
    """

    return "PLANNER"  # Stub; replace with llm.invoke(prompt).

def call_llm_to_plan(state: GraphState):
    """Stub for LLM-generated analysis plan."""

    prompt = f"""
    You are a data analyst. Produce a concise analysis plan.

    User request: {state['user_request']}
    Dataset profile: {state.get('dataset_profile', 'not available')}

    Rules:
    1. If the task is trivial (e.g. row count, column names), reply only "DIRECT".
    2. Otherwise list logical steps (at most 5).
    3. Each step starts with a number, e.g. "1. [step content]".
    """
    return "1. Check missing values\n2. Compute GPA mean\n3. Run t-test"


def parse_plan(text: str) -> list:
    """Parse numbered steps from LLM text."""
    if "DIRECT" in text.upper():
        return ["Execute the user instruction directly"]
    steps = re.findall(r'\d\.\s*(.*)', text)
    return steps if steps else [text]

def planner_node(state: GraphState):
    """Planner: derive analysis_plan from profile (stub)."""
    if state.get("analysis_plan"):
        return {}

    response = call_llm_to_plan(state)

    if "DIRECT" in response:
        return {"analysis_plan": ["Answer the user directly"]}

    plan_steps = parse_plan(response)
    return {"analysis_plan": plan_steps}


def router_gate(state: GraphState):
    """
    Current architecture decision:
    all analytical decisions go through the Supervisor.

    The old planner_node is a historical stub and must not drive execution.
    We keep the node in the graph for compatibility, but the active route is
    always build_context -> supervisor.
    """
    return "supervisor"

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
    """
    Soft answer-quality gate for the workbench-style analyst loop.

    This replaces the old task_contract-centered deliverable behavior on the
    active path. It checks whether the final answer is grounded in recorded
    analysis evidence, current data version, and visible limitations.

    It intentionally does not force rigid plan/workflow completion.
    """
    observations = state.get("observations", []) or []
    analysis_runs = state.get("analysis_runs", []) or []

    check = check_answer_quality(
        user_request=state.get("user_request", ""),
        current_action=state.get("current_action"),
        analysis_runs=analysis_runs,
        observations=observations,
        active_data_version_id=state.get("active_data_version_id"),
    )

    gate_attempts = int(state.get("deliverable_gate_attempts", 0)) + 1

    print("\n" + "=" * 40)
    print("[ANSWER QUALITY GATE]")
    print(json.dumps(check, ensure_ascii=False, indent=2)[:4000])
    print("=" * 40 + "\n")

    return {
        "deliverable_check": check,
        "deliverable_gate_attempts": gate_attempts,
    }

def route_after_deliverable_gate(state: GraphState):
    """
    Route after the soft answer-quality gate.

    The active architecture uses this gate as a final quality check, not as a
    rigid workflow loop. The gate records warnings but normally lets the answer
    finish. Legacy task-contract checks can still use missing/blocked if they
    are reintroduced explicitly in the future.
    """
    check = state.get("deliverable_check") or {}
    status = check.get("status")
    gate_type = check.get("gate_type")

    print(f"[ROUTE AFTER ANSWER QUALITY GATE] status = {status}, gate_type = {gate_type}")

    if gate_type == "answer_quality_gate":
        return "end"

    if status == "ok":
        return "end"

    if status == "blocked":
        return "end"

    return "build_context"

# --- Compile graph ---
workflow = StateGraph(GraphState)

workflow.add_node("build_context", build_context_node)
workflow.add_node("planner", planner_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("verify", verify_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("execute", execute_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("deliverable_gate", deliverable_gate_node)

workflow.set_entry_point("build_context")

# Supervisor loop
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
        "end": END,
        "build_context": "build_context",
    }
)

workflow.add_conditional_edges(
    "verify",
    route_after_verify,
    {
        "execute": "execute",
        "human_review": "human_review",
        "build_context": "build_context",
    },
)

# After build_context: optional planner
workflow.add_conditional_edges(
    "build_context",
    router_gate,
    {
        "planner": "planner",
        "supervisor": "supervisor"
    }
)

# Planner hands off to supervisor
workflow.add_edge("planner", "supervisor")

workflow.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "execute": "execute",
        "build_context": "build_context",
    }
)

workflow.add_edge("execute", "summarize")

# After summarize, loop back to build_context
workflow.add_conditional_edges(
    "summarize",
    route_after_summarize,
    {
        "build_context": "build_context",
        "end": END,
    }
)

# Compile with checkpoint + interrupt
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])