from pathlib import Path


def test_dataset_profile_is_serialized_before_entering_graph_state():
    upload_text = Path("core/app_backend/dataset_upload.py").read_text(
        encoding="utf-8"
    )
    context_text = Path("core/workflow/nodes/context.py").read_text(
        encoding="utf-8"
    )

    assert '"dataset_profile": state_dataset_profile,' not in upload_text
    assert '"dataset_profile": state_dataset_profile.model_dump()' in upload_text

    assert '"dataset_profile": new_profile,' not in context_text
    assert '"dataset_profile": new_profile.model_dump()' in context_text


def test_supervisor_serializes_action_before_state_update():
    text = Path("core/workflow/nodes/supervisor.py").read_text(
        encoding="utf-8"
    )

    assert "action_to_state_dict" in text
    assert 'updates = {"current_action": action}' not in text
    assert 'updates = {"current_action": action_to_state_dict(action)}' in text


def test_plan_execution_serializes_action_before_state_update():
    text = Path("core/workflow/nodes/plan_execution.py").read_text(
        encoding="utf-8"
    )

    assert "action_to_state_dict" in text
    assert '"current_action": action,' not in text
    assert '"current_action": action_to_state_dict(action),' in text


def test_verification_node_serializes_verification_before_state_update():
    text = Path("core/workflow/nodes/verification.py").read_text(
        encoding="utf-8"
    )

    assert "verification_to_state_dict" in text
    assert '"current_verification": verify_result' not in text
    assert '"current_verification": verification_payload' in text


def test_human_review_node_serializes_action_and_verification_before_state_update():
    text = Path("core/workflow/nodes/human_review.py").read_text(
        encoding="utf-8"
    )

    assert "action_to_state_dict" in text
    assert "verification_to_state_dict" in text

    assert '"current_action": action,' not in text
    assert '"current_verification": vr,' not in text
    assert '"current_verification": approved_vr,' not in text


def test_validator_uses_action_access_helpers_for_dict_actions():
    text = Path("core/analysis_tool_plugins/validation.py").read_text(
        encoding="utf-8"
    )

    assert "get_action_tool_name" in text
    assert "get_action_arguments" in text
    assert 'getattr(action, "tool_name", None)' not in text
    assert 'getattr(action, "arguments", {})' not in text


def test_execution_layer_uses_action_access_helpers_for_dict_actions():
    text = Path("core/analysis_tool_plugins/execution.py").read_text(
        encoding="utf-8"
    )

    assert "get_action_tool_name" in text
    assert "get_action_arguments" in text
    assert "get_action_id" in text

    assert "action.tool_name" not in text
    assert "action.arguments" not in text
    assert "action.action_id" not in text