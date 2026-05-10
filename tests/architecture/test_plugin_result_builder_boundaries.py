from pathlib import Path


def test_guardrail_evaluation_lives_inside_result_builder():
    result_builder = Path(
        "core/analysis_tool_plugins/result_builder.py"
    ).read_text(encoding="utf-8")

    assert "def evaluate_guardrails_for_plugin" in result_builder
    assert "core.analysis_tool_plugins.guardrails" not in result_builder


def test_standalone_guardrails_module_removed():
    assert not Path("core/analysis_tool_plugins/guardrails.py").exists()