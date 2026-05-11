from pathlib import Path


def test_graph_state_is_defined_only_in_core_state():
    target = "class " + "GraphState"

    definitions = []

    for path in Path(".").rglob("*.py"):
        normalized = str(path).replace("\\", "/")

        if any(part in path.parts for part in {".git", "venv", ".venv", "__pycache__"}):
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        if target in text:
            definitions.append(normalized)

    assert definitions == ["core/state.py"]