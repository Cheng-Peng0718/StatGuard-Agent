from core.deliverables.contracts import TaskContract, normalize_task_contract


def test_normalize_empty_contract():
    contract = normalize_task_contract(None)

    assert isinstance(contract, TaskContract)
    assert contract.required_tools == []
    assert contract.required_artifacts == []
    assert contract.required_deliverables == []
    assert contract.success_criteria == []
    assert contract.allow_partial is False


def test_normalize_legacy_task_contract_dict():
    contract = normalize_task_contract({
        "required_tools": "get_summary_stats",
        "required_artifacts": ["plot"],
        "required_deliverables": ("summary", "limitations"),
        "success_criteria": "mention missingness",
        "allow_partial": True,
        "custom_field": "kept",
    })

    assert contract.required_tools == ["get_summary_stats"]
    assert contract.required_artifacts == ["plot"]
    assert contract.required_deliverables == ["summary", "limitations"]
    assert contract.success_criteria == ["mention missingness"]
    assert contract.allow_partial is True
    assert contract.metadata["custom_field"] == "kept"