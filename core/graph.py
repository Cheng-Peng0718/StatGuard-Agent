import os
from verifiers.validators import verify
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from core.state import GraphState

from core.schema import Observation
from core.context_builder import build_context, generate_profile

from core.data_versions import get_active_data_path

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

from core.verification_access import (
    get_verification_details,
    get_verification_error_code,
    get_verification_feedback,
    get_verification_status,
    set_verification_fields,
)

from core.workflow.repair_runtime import attach_repair_decision

from core.workflow.runtime_utils import get_action_hash

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

from core.workflow.verification_feedback import attach_verification_blocked_response

from core.workflow.nodes.execution import execute_node

from core.workflow.nodes.summarization import summarize_node

from core.workflow.nodes.human_review import human_review_node

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
        updates = attach_verification_blocked_response(state, updates)
        updates.update(clear_after_repair)

        return updates

    updates = {
        "current_verification": verify_result,
        "human_review_required": False,
    }

    return attach_repair_decision(state, updates)

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