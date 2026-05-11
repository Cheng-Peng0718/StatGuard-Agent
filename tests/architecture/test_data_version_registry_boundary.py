from pathlib import Path


def test_graph_does_not_define_data_version_registry_helpers():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    forbidden_defs = [
        "def create_initial_data_version",
        "def create_child_data_version",
        "def make_audit_event",
        "def get_active_data_path",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text


def test_data_version_registry_helpers_live_in_data_versions_module():
    text = Path("core/data_versions.py").read_text(encoding="utf-8")

    required_defs = [
        "def create_initial_data_version",
        "def create_child_data_version",
        "def make_audit_event",
        "def get_active_data_path",
        "def extract_data_version_update",
        "def validate_data_version_update",
    ]

    for required in required_defs:
        assert required in text