from pathlib import Path


def test_workflow_profile_access_helpers_exist():
    text = Path("core/workflow/profile_access.py").read_text(encoding="utf-8")

    assert "def get_context_profile" in text
    assert "def require_context_profile" in text
    assert "def get_dataset_profile_v2" in text
    assert "def require_dataset_profile_v2" in text


def test_workflow_nodes_use_profile_access_boundary():
    checked_files = [
        "core/workflow/nodes/execution.py",
        "core/workflow/nodes/supervisor.py",
        "core/workflow/nodes/verification.py",
        "core/workflow/nodes/plan_execution.py",
    ]

    for path in checked_files:
        text = Path(path).read_text(encoding="utf-8")
        assert "core.workflow.profile_access" in text


def test_active_workflow_nodes_do_not_directly_fetch_dataset_profile_state():
    forbidden_patterns = {
        "core/workflow/nodes/execution.py": [
            'state.get("dataset_profile")',
            "state.get('dataset_profile')",
        ],
        "core/workflow/nodes/supervisor.py": [
            'state.get("dataset_profile")',
            "state.get('dataset_profile')",
        ],
        "core/workflow/nodes/verification.py": [
            'state["dataset_profile"]',
            "state['dataset_profile']",
        ],
        "core/workflow/nodes/plan_execution.py": [
            'state.get("dataset_profile")',
            "state.get('dataset_profile')",
            'state.get("dataset_profile_v2")',
            "state.get('dataset_profile_v2')",
        ],
    }

    for path, patterns in forbidden_patterns.items():
        text = Path(path).read_text(encoding="utf-8")
        for pattern in patterns:
            assert pattern not in text

def test_verification_and_plan_execution_use_dataset_profile_v2_for_validation():
    verification_text = Path(
        "core/workflow/nodes/verification.py"
    ).read_text(encoding="utf-8")
    plan_execution_text = Path(
        "core/workflow/nodes/plan_execution.py"
    ).read_text(encoding="utf-8")

    assert "require_dataset_profile_v2" in verification_text
    assert "require_context_profile" not in verification_text

    assert "get_dataset_profile_v2" in plan_execution_text
    assert "get_context_profile" not in plan_execution_text