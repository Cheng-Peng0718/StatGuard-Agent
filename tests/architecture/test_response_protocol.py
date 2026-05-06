from pathlib import Path


def test_new_response_nodes_do_not_return_final_answer_directly():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    for fn_name in [
        "advisory_answer_node",
        "plan_only_node",
        "execute_pending_plan_node",
    ]:
        start = graph_text.index(f"def {fn_name}")
        # Stop at next function definition after this one.
        rest = graph_text[start + 1:]
        next_def_offset = rest.find("\ndef ")
        body = graph_text[start:] if next_def_offset == -1 else graph_text[start:start + 1 + next_def_offset]

        assert '"final_answer"' not in body
        assert "'final_answer'" not in body
