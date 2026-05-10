from pathlib import Path


def test_analysis_runs_does_not_use_legacy_migration_language():
    text = Path("core/analysis_runs.py").read_text(encoding="utf-8")

    forbidden = [
        "_generic_unified_fallback_plugin",
        "Existing graph compatibility",
        "legacy analysis plugin",
        "unified fallback",
        "after migration",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_analysis_runs_keeps_unknown_tool_placeholder_boundary():
    text = Path("core/analysis_runs.py").read_text(encoding="utf-8")

    assert "def _generic_placeholder_plugin" in text
    assert "Unknown tools use a generic placeholder plugin" in text