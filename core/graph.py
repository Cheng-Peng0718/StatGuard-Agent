
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from core.state import GraphState

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

from core.workflow.nodes.context import build_context_node


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
        "end": END,
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