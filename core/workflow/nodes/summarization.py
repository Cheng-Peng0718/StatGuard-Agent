from __future__ import annotations

import uuid

from core.action_access import (
    get_action_arguments,
    get_action_id,
    get_action_tool_name,
)
from core.analysis_runs import build_analysis_run_from_observation
from core.data_versions import (
    extract_data_version_update,
    validate_data_version_update,
)
from core.execution_codec import normalize_execution_view
from core.planning.execution_queue import mark_plan_step_after_execution
from core.workflow.audit_runtime import attach_execution_audit
from core.workflow.repair_runtime import attach_repair_after_summarize


def summarize_node(state: dict):
    current_action = state.get("current_action")
    tool_name = get_action_tool_name(current_action, "unknown_tool")
    arguments = get_action_arguments(current_action)

    raw_result = state.get("current_execution", "No execution result")

    execution_view = normalize_execution_view(
        raw_result,
        fallback_action_id=get_action_id(current_action),
        fallback_tool_name=tool_name,
    )

    execution_id = execution_view.get("execution_id")
    status = execution_view.get("status")
    success = bool(execution_view.get("success"))
    error_code = execution_view.get("error_code")
    message = execution_view.get("message")
    artifacts = execution_view.get("artifacts") or []
    payload = execution_view.get("payload") or {}
    raw_result = execution_view

    summary = (
        f"Tool {tool_name} finished with status={status}, success={success}. "
        f"message={message or 'No message'}"
    )

    if error_code:
        summary += f" error_code={error_code}."

    refined_observation = {
        "observation_id": f"obs_{uuid.uuid4().hex[:8]}",
        "source_action_id": get_action_id(current_action),
        "tool_name": tool_name,
        "arguments": arguments,

        # Provenance
        "data_version_id": state.get("active_data_version_id"),

        "status": status,
        "success": success,
        "error_code": error_code,
        "message": message,
        "artifacts": artifacts,
        "summary": summary,
        "structured_data": {
            "status": status,
            "success": success,
            "error_code": error_code,
            "message": message,
            "artifacts": artifacts,
            "payload": payload,
            "data_version_id": state.get("active_data_version_id"),
        },
        "raw_data": raw_result,
    }

    print(f"[Summarize]: archived result for {tool_name}.")

    updates = {
        "observations": [refined_observation],

        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,

        "current_step": state.get("current_step", 0) + 1,
    }

    data_version_update = extract_data_version_update(raw_result)
    validated_version_update = validate_data_version_update(data_version_update)

    if validated_version_update is not None:
        new_version = validated_version_update["new_version"]
        new_active_id = validated_version_update["active_data_version_id"]
        audit_event = validated_version_update.get("audit_event")

        existing_versions = state.get("data_versions", []) or []
        existing_audit_log = state.get("data_audit_log", []) or []

        updates["data_versions"] = existing_versions + [new_version]
        updates["active_data_version_id"] = new_active_id

        if audit_event:
            updates["data_audit_log"] = existing_audit_log + [audit_event]

        refined_observation["data_version_id"] = new_active_id

        structured_data = refined_observation.get("structured_data")
        if isinstance(structured_data, dict):
            structured_data["data_version_id"] = new_active_id

        if isinstance(payload, dict):
            payload["active_data_version_id"] = new_active_id
            payload["data_version_id"] = new_active_id

        print(f"[DATA VERSION] active_data_version_id -> {new_active_id}")

    if tool_name not in {"unknown_tool"}:
        analysis_run = build_analysis_run_from_observation(
            observation=refined_observation,
        )

        # Important: this is a delta. Do not change to existing + [analysis_run].
        updates["analysis_runs"] = [analysis_run]

    current_plan_step_id = state.get("current_plan_step_id")
    pending_plan = state.get("pending_plan")

    if current_plan_step_id and pending_plan:
        updated_plan = mark_plan_step_after_execution(
            pending_plan,
            step_id=current_plan_step_id,
            success=success,
            execution_id=execution_id,
            message=message,
        )

        updates["pending_plan"] = updated_plan
        updates["plan_status"] = updated_plan.get("status")
        updates["current_plan_step_id"] = None

        updates["action_origin"] = None

        print(
            f"[PLAN EXECUTION] step {current_plan_step_id} "
            f"marked as {'completed' if success else 'failed'}"
        )

    updates = attach_repair_after_summarize(
        state=state,
        updates=updates,
        current_action=current_action,
        raw_result=raw_result,
        tool_name=tool_name,
    )

    return attach_execution_audit(state, updates)