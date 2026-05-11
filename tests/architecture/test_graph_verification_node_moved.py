from pathlib import Path


def test_verify_node_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    verification_text = Path("core/workflow/nodes/verification.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def verify_node" not in graph_text
    assert "def verify_node" in verification_text


def test_core_graph_imports_verify_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert (
        "from core.workflow.nodes.verification import verify_node"
        in graph_text
    )

    forbidden_imports = [
        "from verifiers.validators import verify",
        "from core.schema import Observation",
        "from core.workflow.repair_runtime import attach_repair_decision",
        "from core.workflow.verification_feedback import attach_verification_blocked_response",
        "from core.workflow.runtime_utils import get_action_hash",
        "mark_plan_step_after_execution",
        "set_verification_fields",
        "get_verification_status",
        "get_verification_feedback",
        "get_verification_error_code",
        "get_verification_details",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_text


def test_verification_node_keeps_rejection_repair_response_order():
    verification_text = Path("core/workflow/nodes/verification.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    repair_idx = verification_text.index("updates = attach_repair_decision")
    response_idx = verification_text.index("updates = attach_verification_blocked_response")
    clear_idx = verification_text.index("updates.update(clear_after_repair)")

    assert repair_idx < response_idx < clear_idx