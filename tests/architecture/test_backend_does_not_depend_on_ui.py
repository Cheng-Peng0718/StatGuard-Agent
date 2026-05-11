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

def test_core_backend_does_not_import_streamlit_or_app():
    offenders = []

    for path in Path("core").rglob("*.py"):
        offenders.extend(_find_forbidden_imports(path))

    assert offenders == []


def test_backend_tests_do_not_import_streamlit_or_app_except_ui_legacy_tests():
    offenders = []

    for path in Path("tests").rglob("*.py"):
        normalized = path.as_posix()

        if "/ui_legacy/" in normalized or "/ui/" in normalized:
            continue

        offenders.extend(_find_forbidden_imports(path))

    assert offenders == []


def test_ui_adapter_is_the_only_core_ui_boundary_package():
    ui_adapter_path = Path("core/ui_adapter")

    assert ui_adapter_path.exists()
    assert (ui_adapter_path / "events.py").exists()
    assert (ui_adapter_path / "snapshot.py").exists()