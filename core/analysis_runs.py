from typing import Any, Dict, List, Optional

from core.analysis_plugins import get_plugin


def build_analysis_run_from_observation(
    *,
    tool_name: str,
    action_id: str,
    arguments: Dict[str, Any],
    data_version_id: Optional[str],
    status: str,
    success: bool,
    message: Optional[str],
    payload: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    observation_id: str,
) -> Dict[str, Any]:
    """
    Convert one tool observation into a UI-friendly AnalysisRun.

    This dispatcher is method-agnostic.
    All method-specific extraction and guardrails live in AnalysisPlugin objects.
    """
    plugin = get_plugin(tool_name)

    return plugin.build_analysis_run(
        action_id=action_id,
        arguments=arguments or {},
        data_version_id=data_version_id,
        status=status,
        success=success,
        message=message,
        payload=payload or {},
        artifacts=artifacts or [],
        observation_id=observation_id,
    )