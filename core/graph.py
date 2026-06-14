import os
from verifiers.validators import verify
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
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
from agents.coverage_brief import call_coverage_brief
from core.guardrails import evaluate_multiple_comparison_guardrails

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

    # Ensure there is an active data version even when the dataset arrived via
    # the fallback working_data path (which never went through
    # create_initial_data_version). The id is derived from the FILE CONTENT, so:
    #   - pure analysis tools (which do not change the data) keep the same id,
    #     letting the supervisor reuse prior results instead of re-running and
    #     hitting the fingerprint gate in a loop;
    #   - a tool that actually rewrites the data produces a new version through
    #     the normal data-version path, so the id changes only when data changes.
    updates_version_state = {}
    active_version_id = state.get("active_data_version_id")
    data_versions = state.get("data_versions", []) or []
    if not active_version_id:
        try:
            with open(current_data_path, "rb") as _fh:
                content_hash = hashlib.md5(_fh.read()).hexdigest()[:8]
        except Exception:
            content_hash = "unknown"
        active_version_id = f"working_{content_hash}"
        # generate_profile returns a DatasetProfile object (not a dict), so read
        # n_rows via getattr; reading it as a dict left every fallback version
        # registered with n_rows=0, which made the supervisor believe the active
        # dataset was empty and retry forever.
        if isinstance(new_profile, dict):
            n_rows_val = int(new_profile.get("n_rows", 0))
            n_cols_val = int(new_profile.get("n_cols", 0))
        else:
            n_rows_val = int(getattr(new_profile, "n_rows", 0) or 0)
            n_cols_val = int(getattr(new_profile, "n_cols", 0) or 0)
        # load_df() reads the active version's path with pd.read_parquet, so the
        # registered path MUST be the parquet file -- not whatever working_data*
        # os.listdir happened to hit first (it may be the .csv, which would make
        # read_parquet fail and cascade into a no-data retry loop).
        parquet_path = os.path.join(current_workspace, "working_data.parquet")
        version_path = parquet_path if os.path.exists(parquet_path) else current_data_path
        # Register the version if not already present.
        if not any(v.get("version_id") == active_version_id for v in data_versions):
            data_versions = data_versions + [{
                "version_id": active_version_id,
                "parent_version_id": None,
                "path": version_path,
                "n_rows": n_rows_val,
                "n_cols": n_cols_val,
                "label": "working_data (content-addressed)",
            }]
        updates_version_state = {
            "active_data_version_id": active_version_id,
            "data_versions": data_versions,
        }

    context = build_context(
        step=step,
        max_steps=state["max_steps"],
        user_request=state["user_request"],
        profile=new_profile,
        observations=state.get("observations", []),
        workspace_dir=state.get("workspace_dir", "./"),
        deliverable_check=state.get("deliverable_check"),
        data_versions=data_versions,
        active_data_version_id=active_version_id,
        data_audit_log=state.get("data_audit_log", []),
        analysis_coverage_brief=state.get("analysis_coverage_brief"),
        analysis_runs=state.get("analysis_runs", []),
    )

    return {
        "current_step": step,
        "current_context_text": context.context_text,
        "dataset_profile": new_profile,
        **updates_version_state,
    }

def coverage_brief_node(state: GraphState):
    """
    Build or reuse an LLM-generated Analysis Coverage Brief.

    The brief says which evidence categories should be covered.
    It is not a workflow plan and does not choose exact tools.
    """
    user_request = state.get("user_request", "") or ""
    request_hash = hashlib.md5(user_request.encode("utf-8")).hexdigest()

    existing_hash = state.get("analysis_coverage_request_hash")
    existing_brief = state.get("analysis_coverage_brief")

    if existing_brief and existing_hash == request_hash:
        return {}

    context_text = state.get("current_context_text", "") or ""

    try:
        brief = call_coverage_brief(
            user_request=user_request,
            context_text=context_text,
        )
    except Exception as exc:
        print(f"[COVERAGE BRIEF] failed: {type(exc).__name__}: {exc}")
        brief = {
            "analysis_goal": "coverage_brief_unavailable",
            "required_evidence_categories": [],
            "required_evidence_counts": {},
            "optional_evidence_categories": [],
            "autonomy_level": "answer_now",
            "reasoning_summary": (
                "Coverage brief generation failed; supervisor should proceed with normal one-action reasoning."
            ),
        }

    print("\n" + "=" * 40)
    print("[ANALYSIS COVERAGE BRIEF]")
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print("=" * 40 + "\n")

    return {
        "analysis_coverage_brief": brief,
        "analysis_coverage_request_hash": request_hash,
        "answer_quality_continuation_attempts": 0,
    }

def supervisor_node(state: GraphState):
    current_workspace = state.get("workspace_dir", "./")
    current_profile = state.get("dataset_profile")

    # Build the statistical claims catalogue from all completed runs, so the
    # supervisor can reference claims by ID instead of authoring statistics.
    from core.analysis_tool_plugins.shared.claims_builders import build_claims_for_run
    from core.claims import ClaimSet
    _claimset = ClaimSet()
    for _run in state.get("analysis_runs", []) or []:
        _claimset.add_many(build_claims_for_run(_run))
    _claims_catalogue = _claimset.catalogue_text() if not _claimset.is_empty() else None

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
        analysis_coverage_brief=state.get("analysis_coverage_brief"),
        analysis_runs=state.get("analysis_runs", []),
        claims_catalogue=_claims_catalogue,
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
        analysis_coverage_brief=state.get("analysis_coverage_brief"),
        analysis_runs=state.get("analysis_runs", []),
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
    analysis_runs = state.get("analysis_runs", []) or []\


    check = check_answer_quality(
        user_request=state.get("user_request", ""),
        current_action=state.get("current_action"),
        analysis_runs=analysis_runs,
        observations=observations,
        active_data_version_id=state.get("active_data_version_id"),
        analysis_coverage_brief=state.get("analysis_coverage_brief"),
    )
    # Session-level guardrails run here because they span multiple plugins
    # and cannot be expressed by any single plugin's evaluator. Attachment
    # logic is encapsulated in the evaluator itself.
    # NOTE: analysis_runs MUST be passed as the keyword arg; passing it
    # positionally lands it in `context_or_run` and silently yields no findings.
    session_findings = evaluate_multiple_comparison_guardrails(
        None, analysis_runs=analysis_runs
    )

    if session_findings:
        print(f"[SESSION GUARDRAIL] {len(session_findings)} session-level finding(s)")

    # --- Statistical claims: substitute [CLAIM:id] refs in the final answer
    # with verified wording, and detect bare (unbound) statistical assertions.
    from core.analysis_tool_plugins.shared.claims_builders import build_claims_for_run
    from core.claims import ClaimSet, Claim, CLAIM_SESSION_WARNING

    claimset = ClaimSet()
    for _run in analysis_runs:
        claimset.add_many(build_claims_for_run(_run))

    # Expose the multiple-comparison warning as a citable claim too.
    for i, finding in enumerate(session_findings or []):
        claimset.add(Claim(
            claim_id=f"session_mc_{i}",
            kind=CLAIM_SESSION_WARNING,
            subject="multiple comparisons",
            data={"text": finding.get("message") or finding.get("title", "")},
        ))

    claims_validation = None
    action = state.get("current_action")
    action_type = getattr(action, "action_type", None)
    if action is not None and action_type == "final_answer" and not claimset.is_empty():
        raw_answer = getattr(action, "reasoning_summary", "") or ""
        claims_validation = claimset.validate(raw_answer)
        rendered_answer, unresolved = claimset.substitute(raw_answer)
        # Append the session-level multiple-comparison warning if it was not
        # already cited by the model.
        if session_findings and not any(
                cid.startswith("session_mc_") for cid in claims_validation.get("referenced_ids", [])
        ):
            warning_text = "; ".join(
                f.get("title", "") for f in session_findings if f.get("title")
            )
            if warning_text:
                rendered_answer += f"\n\n_Statistical note: {warning_text}._"
        # Write the rendered answer back so the UI shows verified wording.
        try:
            action.reasoning_summary = rendered_answer
        except Exception:
            pass
        if claims_validation and not claims_validation.get("is_clean"):
            print(f"[CLAIMS] validation flagged: {claims_validation}")

    gate_attempts = int(state.get("deliverable_gate_attempts", 0)) + 1

    continuation_attempts = int(state.get("answer_quality_continuation_attempts", 0))

    if check.get("continuation_recommended"):
        continuation_attempts += 1
    else:
        continuation_attempts = 0

    print("\n" + "=" * 40)
    print("[ANSWER QUALITY GATE]")
    print(json.dumps(check, ensure_ascii=False, indent=2)[:4000])
    print("=" * 40 + "\n")

    return {
        "deliverable_check": check,
        "deliverable_gate_attempts": gate_attempts,
        "answer_quality_continuation_attempts": continuation_attempts,
        "current_action": state.get("current_action"),
        "claims_validation": claims_validation,
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
        continuation_recommended = bool(check.get("continuation_recommended"))
        continuation_attempts = int(state.get("answer_quality_continuation_attempts", 0))
        max_continuation_attempts = 5

        if continuation_recommended and continuation_attempts <= max_continuation_attempts:
            print(
                "[ROUTE AFTER ANSWER QUALITY GATE] continuation recommended; "
                f"attempt {continuation_attempts}/{max_continuation_attempts} -> build_context"
            )
            return "build_context"

        return "end"

    if status == "ok":
        return "end"

    if status == "blocked":
        return "end"

    return "build_context"

# --- Compile graph ---
workflow = StateGraph(GraphState)

workflow.add_node("build_context", build_context_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("verify", verify_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("execute", execute_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("coverage_brief", coverage_brief_node)
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

# All analytical decisions go through the Supervisor. build_context always
# hands off to coverage_brief -> supervisor. The historical planner_node and
# its plan->execute path were removed; do NOT reintroduce a "planner" route
# here. Guarded by tests/architecture/test_no_planner_node.py.
workflow.add_edge("build_context", "coverage_brief")

workflow.add_edge("coverage_brief", "supervisor")

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