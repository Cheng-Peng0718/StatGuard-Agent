from pathlib import Path


def test_audit_runtime_helpers_live_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    audit_text = Path("core/workflow/audit_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def _as_plain_dict",
        "def _merge_state_for_audit",
        "def _compact_state_serialization_audit",
        "def _attach_state_serialization_audit",
        "def _attach_execution_audit",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def as_plain_dict",
        "def merge_state_for_audit",
        "def compact_state_serialization_audit",
        "def attach_state_serialization_audit",
        "def attach_execution_audit",
    ]

    for required in required_defs:
        assert required in audit_text


def test_core_graph_imports_only_audit_runtime_boundary_helper():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.audit_runtime import attach_execution_audit" in graph_text

    forbidden_imported_helpers = [
        "as_plain_dict",
        "merge_state_for_audit",
        "compact_state_serialization_audit",
        "attach_state_serialization_audit",
    ]

    import_line = "from core.workflow.audit_runtime import"
    audit_import_section = graph_text[
        graph_text.find(import_line): graph_text.find(import_line) + 300
    ]

    for helper in forbidden_imported_helpers:
        assert helper not in audit_import_section