from typing import Any, Dict, List, Optional

from core.analysis_tool_plugins import get_plugin as get_unified_plugin
from core.analysis_tool_plugins.base import AnalysisToolPlugin


def _generic_unified_fallback_plugin(tool_name: str) -> AnalysisToolPlugin:
    """
    Generic fallback for unknown tools.

    This prevents the report pipeline from crashing if a tool has no
    registered unified plugin. It should be rare after migration.
    """
    return AnalysisToolPlugin(
        tool_name=tool_name,
        display_name=tool_name.replace("_", " ").title(),
    )


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

    Final architecture:
    - Unified AnalysisToolPlugin is the primary path.
    - Unknown tools use a generic unified fallback.
    - No dependency on the legacy analysis plugin package.
    """
    plugin = get_unified_plugin(tool_name)

    if plugin is None:
        plugin = _generic_unified_fallback_plugin(tool_name)

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