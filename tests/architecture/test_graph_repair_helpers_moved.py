from pathlib import Path


def test_repair_runtime_helpers_live_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    repair_text = Path("core/workflow/repair_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def _attach_repair_decision",
        "def _attach_repair_proposal",
        "def _attach_repair_attempt_if_allowed",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def attach_repair_decision",
        "def attach_repair_proposal",
        "def attach_repair_attempt_if_allowed",
        "def attach_repair_after_summarize",
    ]

    for required in required_defs:
        assert required in repair_text


def test_core_graph_imports_only_repair_runtime_boundary_helpers():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.repair_runtime import" in graph_text
    assert "attach_repair_decision" in graph_text
    assert "attach_repair_after_summarize" in graph_text

    forbidden_imports = [
        "evaluate_repair_decision",
        "generate_repair_proposal",
        "append_repair_attempt",
        "can_attempt_repair",
        "make_repair_attempt",
        "attach_repair_proposal",
        "attach_repair_attempt_if_allowed",
    ]

    graph_import_area = graph_text[:2500]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_import_area