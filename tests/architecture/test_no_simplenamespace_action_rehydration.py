from pathlib import Path


def test_backend_controller_does_not_rehydrate_actions_with_simplenamespace():
    text = Path("core/controller/backend_turn.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_patterns = [
        "from types import SimpleNamespace",
        "types.SimpleNamespace",
        "SimpleNamespace(",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in text