from pathlib import Path


def test_graph_does_not_define_data_version_update_contract_helpers():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    assert "def _extract_data_version_update" not in graph_text
    assert "def _validate_data_version_update" not in graph_text


def test_graph_uses_data_versions_contract_helpers():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    assert "extract_data_version_update" in graph_text
    assert "validate_data_version_update" in graph_text