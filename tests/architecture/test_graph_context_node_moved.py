from pathlib import Path


def test_build_context_node_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    context_text = Path("core/workflow/nodes/context.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def build_context_node" not in graph_text
    assert "def _load_dataframe_for_dataset_intelligence" not in graph_text

    assert "def build_context_node" in context_text
    assert "def _load_dataframe_for_dataset_intelligence" in context_text


def test_core_graph_imports_build_context_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.nodes.context import build_context_node" in graph_text

    forbidden_imports = [
        "from core.context_builder import build_context, generate_profile",
        "from core.data_versions import get_active_data_path",
        "from core.dataset_intelligence.profiler import profile_dataframe, summarize_profile",
        "from core.dataset_intelligence.capability_map import build_capability_map",
        "import pandas as pd",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_text