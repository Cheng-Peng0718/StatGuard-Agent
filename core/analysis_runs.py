from typing import Any, Dict, List, Optional

from core.analysis_tool_plugins import get_plugin as get_unified_plugin
from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.result_builder import build_analysis_run_for_plugin


def _generic_placeholder_plugin(tool_name: str) -> AnalysisToolPlugin:
    """
    Generic placeholder for unknown tools.

    This prevents the report pipeline from crashing if an observation
    references a tool that is not currently registered.
    """
    return AnalysisToolPlugin(
        tool_name=tool_name,
        display_name=tool_name.replace("_", " ").title(),
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def build_analysis_run_from_observation(
    observation: Optional[Any] = None,
    *,
    tool_name: Optional[str] = None,
    action_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    data_version_id: Optional[str] = None,
    status: Optional[str] = None,
    success: Optional[bool] = None,
    message: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    observation_id: Optional[str] = None,
    error_code: Optional[str] = None,
    summary: Optional[str] = None,
    structured_data: Optional[Dict[str, Any]] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Convert one tool Observation into a standardized AnalysisRun.

    Supported inputs:
    1. Observation object or dict:
        build_analysis_run_from_observation(observation=obs)

    2. Explicit fields:
        build_analysis_run_from_observation(
            tool_name=...,
            action_id=...,
            arguments=...,
            ...
        )

    Architecture:
    - AnalysisToolPlugin is the primary renderer/extractor path.
    - This function is the normalization boundary from Observation to AnalysisRun.
    - Unknown tools use a generic placeholder plugin so reporting remains robust.
    """
    obs = _as_dict(observation)

    if observation is not None:
        tool_name = tool_name or obs.get("tool_name")
        action_id = action_id or obs.get("action_id") or obs.get("source_action_id")
        arguments = arguments if arguments is not None else (obs.get("arguments") or {})
        data_version_id = data_version_id or obs.get("data_version_id")
        status = status or obs.get("status")
        success = success if success is not None else obs.get("success")
        error_code = error_code if error_code is not None else obs.get("error_code")
        message = message if message is not None else obs.get("message")
        summary = summary if summary is not None else obs.get("summary")
        artifacts = artifacts if artifacts is not None else (obs.get("artifacts") or [])
        observation_id = observation_id or obs.get("observation_id") or obs.get("id")

        structured_data = (
            structured_data
            if structured_data is not None
            else (obs.get("structured_data") or {})
        )

        raw_data = (
            raw_data
            if raw_data is not None
            else (obs.get("raw_data") or {})
        )

        guardrails = (
            guardrails
            if guardrails is not None
            else (obs.get("guardrails") or [])
        )

        # Existing plugin renderers expect a payload. For Observation-based calls,
        # structured_data is the safest normalized payload source.
        if payload is None:
            payload = obs.get("payload") or structured_data or {}

    arguments = arguments or {}
    payload = payload or {}
    artifacts = artifacts or []
    structured_data = structured_data if structured_data is not None else payload
    raw_data = raw_data or {}
    guardrails = guardrails or []

    if success is None:
        if status in {"ok", "success", "completed", "warning"}:
            success = True
        elif status in {"failed", "error", "rejected"}:
            success = False

    if status is None:
        status = "ok" if success is True else "failed" if success is False else "unknown"

    if summary is None:
        summary = message

    if not tool_name:
        raise ValueError("Cannot build AnalysisRun: missing tool_name.")

    if not observation_id:
        raise ValueError("Cannot build AnalysisRun: missing observation_id.")

    plugin = get_unified_plugin(tool_name)

    if plugin is None:
        plugin = _generic_placeholder_plugin(tool_name)

    try:
        run = build_analysis_run_for_plugin(
            plugin,
            action_id=action_id,
            arguments=arguments,
            data_version_id=data_version_id,
            status=status,
            success=success,
            message=message,
            payload=payload,
            artifacts=artifacts,
            observation_id=observation_id,
        )
        run = _as_dict(run)
    except Exception as exc:
        # Failed or malformed tool outputs should still become AnalysisRun records.
        # The canonical fields below will preserve the real failure status.
        run = {
            "analysis_run_renderer_error": str(exc),
        }

    # Canonical AnalysisRun contract fields.
    # These fields must be stable regardless of plugin-specific renderers.
    run["observation_id"] = observation_id
    run["tool_name"] = tool_name
    run["action_id"] = action_id
    run["arguments"] = arguments
    run["status"] = status
    run["success"] = success
    run["error_code"] = error_code
    run["message"] = message
    plugin_summary = run.get("summary")

    run["plugin_summary"] = plugin_summary
    run["summary"] = summary or message or plugin_summary
    run["data_version_id"] = data_version_id
    run["artifacts"] = artifacts
    run["structured_data"] = run.get("structured_data") or structured_data or {}
    run["raw_data"] = run.get("raw_data") or raw_data or {}
    run["guardrails"] = guardrails if guardrails else (run.get("guardrails") or [])

    return run
