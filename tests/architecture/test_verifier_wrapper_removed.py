from pathlib import Path


def test_verify_node_uses_plugin_validator_directly():
    text = Path("core/workflow/nodes/verification.py").read_text(
        encoding="utf-8"
    )

    assert "validate_plugin_action" in text
    assert "verifiers.validators" not in text
    assert "from verifiers" not in text


def test_old_verifiers_validators_wrapper_removed():
    assert not Path("verifiers/validators.py").exists()

def test_old_verifiers_package_removed_if_empty():
    assert not Path("verifiers/__init__.py").exists()