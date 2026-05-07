from pathlib import Path


def test_repair_proposal_module_does_not_execute_tools_or_call_llm():
    text = Path("core/repair/proposal.py").read_text(encoding="utf-8")

    forbidden_fragments = [
        "execute_tool",
        "run_tool",
        "client.chat",
        "responses.create",
        "invoke(",
    ]

    offenders = [
        fragment for fragment in forbidden_fragments
        if fragment in text
    ]

    assert offenders == []


def test_repair_proposal_module_defines_schema_only():
    text = Path("core/repair/proposal.py").read_text(encoding="utf-8")

    assert "class RepairProposal" in text
    assert "def make_argument_repair_proposal" in text
    assert "def make_ask_user_repair_proposal" in text
    assert "def make_method_fallback_repair_proposal" in text
    assert "def make_no_op_repair_proposal" in text