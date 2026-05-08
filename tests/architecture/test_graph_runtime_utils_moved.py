from pathlib import Path


def test_runtime_utils_live_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    runtime_text = Path("core/workflow/runtime_utils.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def sanitize_results",
        "def get_action_hash",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def sanitize_results",
        "def get_action_hash",
    ]

    for required in required_defs:
        assert required in runtime_text


def test_core_graph_imports_runtime_utils_boundary_helpers():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.runtime_utils import sanitize_results, get_action_hash" in graph_text

    forbidden_imports = [
        "import numpy as np",
        "import hashlib",
        "import json",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_text