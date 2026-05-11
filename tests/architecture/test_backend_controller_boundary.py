import ast
from pathlib import Path


def _find_forbidden_imports(path: Path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    offenders = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "app" or name.startswith("app."):
                    offenders.append(f"{path}: import {name}")
                if name == "streamlit" or name.startswith("streamlit."):
                    offenders.append(f"{path}: import {name}")

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "app" or module.startswith("app."):
                offenders.append(f"{path}: from {module}")
            if module == "streamlit" or module.startswith("streamlit."):
                offenders.append(f"{path}: from {module}")

    return offenders


def test_backend_controller_does_not_import_streamlit_or_app():
    offenders = _find_forbidden_imports(
        Path("core/controller/backend_turn.py")
    )

    assert offenders == []


def test_backend_controller_does_not_call_llm_directly():
    text = Path("core/controller/backend_turn.py").read_text(encoding="utf-8")

    forbidden = [
        "client.chat",
        "responses.create",
        "OpenAI(",
        ".invoke(",
    ]

    offenders = [
        fragment
        for fragment in forbidden
        if fragment in text
    ]

    assert offenders == []


def test_backend_controller_exposes_run_backend_turn():
    text = Path("core/controller/backend_turn.py").read_text(encoding="utf-8")

    assert "def run_backend_turn" in text
    assert "build_ui_snapshot" in text
    assert "intent_router_node" in text
    assert "execute_pending_plan_node" in text
    assert "human_review_node" in text