from pathlib import Path


def test_core_graph_is_composition_root_only():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def build_context_node",
        "def verify_node",
        "def human_review_node",
        "def execute_node",
        "def summarize_node",
        "def deliverable_gate_node",
        "def final_response_node",
        "def plan_only_node",
        "def execute_pending_plan_node",
        "def supervisor_node",
        "def advisory_answer_node",
        "def intent_router_node",
        "def route_after_",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    assert "def create_graph_app" in graph_text


def test_core_graph_has_no_business_runtime_imports():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_fragments = [
        "import pandas as pd",
        "import os",
        "from verifiers.validators import verify",
        "from core.context_builder import build_context",
        "generate_profile",
        "profile_dataframe",
        "summarize_profile",
        "build_capability_map",
        "execute_analysis_tool",
        "from core.schema import Observation",
        "make_assistant_response",
        "make_response_update",
        "attach_repair_decision",
        "attach_verification_blocked_response",
        "get_action_hash",
        "get_active_data_path",
    ]

    for forbidden in forbidden_fragments:
        assert forbidden not in graph_text


def test_core_graph_imports_all_nodes_from_workflow_modules():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    required_imports = [
        "from core.workflow.nodes.context import build_context_node",
        "from core.workflow.nodes.interaction import",
        "from core.workflow.nodes.planning import plan_only_node",
        "from core.workflow.nodes.plan_execution import execute_pending_plan_node",
        "from core.workflow.nodes.supervisor import supervisor_node",
        "from core.workflow.nodes.verification import verify_node",
        "from core.workflow.nodes.human_review import human_review_node",
        "from core.workflow.nodes.execution import execute_node",
        "from core.workflow.nodes.summarization import summarize_node",
        "from core.workflow.nodes.finalization import",
        "from core.workflow.routes import",
    ]

    for required in required_imports:
        assert required in graph_text