from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping

from pydantic import BaseModel, Field

from core.analysis_tool_plugins.base import AnalysisToolPlugin

class ToolManifest(BaseModel):
    tool_name: str
    display_name: str
    method_family: str = "general"

    argument_schema: Dict[str, Any] = Field(default_factory=dict)
    variable_roles: List[Dict[str, Any]] = Field(default_factory=list)

    planning_policy: Dict[str, Any] = Field(default_factory=dict)
    versioning_policy: Dict[str, Any] = Field(default_factory=dict)
    repair_policy: Dict[str, Any] = Field(default_factory=dict)

    requires_confirmation: bool = False
    mutates_data: bool = False

    planning_tags: List[str] = Field(default_factory=list)
    supported_goal_types: List[str] = Field(default_factory=list)
    expected_deliverables: List[str] = Field(default_factory=list)
    default_plan_purpose: str = ""
    argument_template: Dict[str, Any] = Field(default_factory=dict)
    task_argument_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    required_planning_choices: List[str] = Field(default_factory=list)
    not_recommended_for_goal_types: List[str] = Field(default_factory=list)
    plan_order: int = 100

    has_applicability_checker: bool = False


def _manifest_value(value: Any) -> Any:
    if isinstance(value, type):
        return value.__name__

    if is_dataclass(value) and not isinstance(value, type):
        return _manifest_value(asdict(value))

    if hasattr(value, "to_contract_dict"):
        return _manifest_value(value.to_contract_dict())

    if hasattr(value, "model_dump"):
        return _manifest_value(value.model_dump())

    if isinstance(value, dict):
        return {
            str(key): _manifest_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_manifest_value(item) for item in value]

    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]

    return value

def _planning_metadata_for_plugin(plugin: AnalysisToolPlugin) -> Dict[str, Any]:
    local_raw = getattr(plugin, "planning_metadata", None)
    local = _manifest_value(local_raw)

    if isinstance(local, dict):
        return local

    return {}

def _planning_plan_order(metadata: Dict[str, Any]) -> int:
    value = metadata.get("plan_order")

    if isinstance(value, int):
        return value

    return 100

def build_tool_manifest(plugin: AnalysisToolPlugin) -> ToolManifest:
    planning_metadata = _planning_metadata_for_plugin(plugin)

    return ToolManifest(
        tool_name=plugin.tool_name,
        display_name=plugin.display_name,
        method_family=plugin.method_family,
        argument_schema=_manifest_value(plugin.argument_schema),
        variable_roles=_manifest_value(plugin.variable_roles),
        planning_policy=_manifest_value(plugin.planning_policy),
        versioning_policy=_manifest_value(plugin.versioning_policy),
        repair_policy=_manifest_value(plugin.repair_policy),
        requires_confirmation=plugin.requires_confirmation,
        mutates_data=plugin.mutates_data,
        planning_tags=_manifest_value(
            planning_metadata.get("planning_tags", [])
        ),
        supported_goal_types=_manifest_value(
            planning_metadata.get("supported_goal_types", [])
        ),
        not_recommended_for_goal_types=_manifest_value(
            planning_metadata.get("not_recommended_for_goal_types", [])
        ),
        expected_deliverables=_manifest_value(
            planning_metadata.get("expected_deliverables", [])
        ),
        default_plan_purpose=planning_metadata.get("default_plan_purpose", ""),
        argument_template=_manifest_value(
            planning_metadata.get("argument_template", {})
        ),
        task_argument_bindings=_manifest_value(
            planning_metadata.get("task_argument_bindings", [])
        ),
        required_planning_choices=_manifest_value(
            planning_metadata.get("required_planning_choices", [])
        ),
        plan_order=_planning_plan_order(planning_metadata),
        has_applicability_checker=plugin.applicability_checker is not None,
    )


def build_tool_manifests(
    plugins: Mapping[str, AnalysisToolPlugin],
) -> Dict[str, ToolManifest]:
    return {
        tool_name: build_tool_manifest(plugin)
        for tool_name, plugin in sorted(plugins.items())
    }
