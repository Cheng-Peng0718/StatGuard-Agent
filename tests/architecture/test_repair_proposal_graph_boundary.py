from pathlib import Path


def test_graph_attaches_repair_proposal_observe_only():
    repair_text = Path("core/workflow/repair_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "generate_repair_proposal" in repair_text
    assert "def attach_repair_proposal" in repair_text


def test_graph_repair_proposal_hook_does_not_execute_tools():
    repair_text = Path("core/workflow/repair_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    start = repair_text.index("def attach_repair_proposal")
    rest = repair_text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    body = (
        repair_text[start:]
        if next_def_offset == -1
        else repair_text[start:start + 1 + next_def_offset]
    )

    for forbidden in [
        "execute_analysis_tool",
        "execute_node",
        "run_backend_turn",
    ]:
        assert forbidden not in body