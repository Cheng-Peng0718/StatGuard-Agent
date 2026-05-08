import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from core.state import GraphState
from core.context_builder import build_context, generate_profile

from core.data_versions import get_active_data_path

from core.dataset_intelligence.profiler import profile_dataframe, summarize_profile
from core.dataset_intelligence.capability_map import build_capability_map

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
from core.workflow.nodes.execution import execute_node

from core.workflow.nodes.summarization import summarize_node

from core.workflow.nodes.human_review import human_review_node

from core.workflow.nodes.verification import verify_node

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