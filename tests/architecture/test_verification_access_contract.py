from pathlib import Path


import ast
from pathlib import Path


def test_core_graph_uses_verification_access_helpers_for_runtime_fields():
    text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(text)

    forbidden_attributes = {
        ("verify_result", "status"),
        ("verify_result", "feedback"),
        ("verify_result", "error_code"),
        ("verify_result", "details"),
        ("approved_vr", "status"),
        ("approved_vr", "feedback"),
    }

    forbidden_getattr_targets = {
        ("verification", "status"),
        ("vr", "status"),
    }

    forbidden_dict_get_targets = {
        ("vr", "status"),
        ("verification", "status"),
    }

    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            pair = (node.value.id, node.attr)
            if pair in forbidden_attributes:
                violations.append(
                    f"{pair[0]}.{pair[1]} at line {node.lineno}"
                )

        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and isinstance(node.args[1], ast.Constant)
            ):
                pair = (node.args[0].id, node.args[1].value)
                if pair in forbidden_getattr_targets:
                    violations.append(
                        f"getattr({pair[0]}, {pair[1]!r}) at line {node.lineno}"
                    )

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
            ):
                pair = (node.func.value.id, node.args[0].value)
                if pair in forbidden_dict_get_targets:
                    violations.append(
                        f"{pair[0]}.get({pair[1]!r}) at line {node.lineno}"
                    )

    assert violations == []


def test_backend_turn_normalizes_verification_at_finish_boundary():
    text = Path("core/controller/backend_turn.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.verification_access import get_verification_status" in text
    assert "from core.verification_codec import verification_to_state_dict" in text
    assert "_VERIFICATION_STATE_FIELDS = (\"current_verification\",)" in text
    assert "def _normalize_state_verifications_for_storage" in text
    assert "state = _normalize_state_verifications_for_storage(state)" in text