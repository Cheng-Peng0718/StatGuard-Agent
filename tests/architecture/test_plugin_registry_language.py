from pathlib import Path


def test_plugin_registry_comments_do_not_reference_legacy_migration_language():
    text = Path("core/analysis_tool_plugins/registry.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "Backward-compatible",
        "backward-compatible",
        "older code",
        "tools.registry",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_plugin_execution_comments_do_not_reference_old_tool_registry():
    text = Path("core/analysis_tool_plugins/execution.py").read_text(
        encoding="utf-8"
    )

    assert "tools.registry" not in text
    assert "fallback to tools" not in text