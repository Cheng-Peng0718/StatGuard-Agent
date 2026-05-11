from pathlib import Path


def _function_body(text: str, fn_name: str) -> str:
    start = text.index(f"def {fn_name}")
    rest = text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    return text[start:] if next_def_offset == -1 else text[start:start + 1 + next_def_offset]


def test_new_response_nodes_do_not_return_final_answer_directly():
    targets = {
        "advisory_answer_node": "core/workflow/nodes/interaction.py",
        "plan_only_node": "core/workflow/nodes/planning.py",
        "execute_pending_plan_node": "core/workflow/nodes/plan_execution.py",
    }

    for fn_name, path in targets.items():
        text = Path(path).read_text(encoding="utf-8")
        body = _function_body(text, fn_name)

        assert '"final_answer"' not in body
        assert "'final_answer'" not in body