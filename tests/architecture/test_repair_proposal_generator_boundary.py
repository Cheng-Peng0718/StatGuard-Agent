from pathlib import Path


def test_repair_proposal_generator_does_not_execute_tools_or_call_llm():
    text = Path("core/repair/proposal_generator.py").read_text(encoding="utf-8")

    forbidden_fragments = [
        "execute_tool",
        "run_tool",
        "client.chat",
        "responses.create",
        ".invoke(",
    ]

    offenders = [
        fragment for fragment in forbidden_fragments
        if fragment in text
    ]

    assert offenders == []


def test_repair_proposal_generator_uses_plugin_contracts():
    text = Path("core/repair/proposal_generator.py").read_text(encoding="utf-8")

    assert "get_plugin" in text
    assert "argument_schema" in text
    assert "value_aliases" in text