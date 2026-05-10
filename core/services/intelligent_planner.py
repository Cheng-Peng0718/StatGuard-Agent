from __future__ import annotations

import uuid
from typing import Dict, Iterable, List, Optional

from core.dataset_intelligence.schemas import (
    AnalysisCapability,
    CapabilityMap,
    DatasetProfileV2,
)
from core.domain.task import TaskSpec
from core.analysis_tool_plugins import PLUGIN_REGISTRY, get_plugin
from core.analysis_tool_plugins.manifest import build_tool_manifest
from core.planning.schemas import PlanProposal, PlanStep
from core.planning.verifier import verify_plan
from core.services.interaction_router import decide_interaction_intent


DATASET_OVERVIEW_TOOLS = [
    "inspect_dataset",
    "missingness_report",
    "get_summary_stats",
    "summarize_columns",
    "get_correlation_matrix",
]

REGRESSION_TOOLS = [
    "run_multiple_regression",
    "regression_diagnostics",
    "generate_residual_histogram",
]

ASSOCIATION_TOOLS = [
    "run_correlation_test",
    "get_correlation_matrix",
]

VISUALIZATION_TOOLS = [
    "generate_scatterplot",
    "generate_residual_histogram",
]

DATA_CLEANING_TOOLS = [
    "clean_data",
]

DISALLOWED_FOR_OVERVIEW = {
    "run_multiple_regression",
    "regression_diagnostics",
    "generate_residual_histogram",
    "run_anova",
    "run_chi_square",
    "run_independent_t_test",
    "clean_data",
}

OVERVIEW_STYLE_GOAL_TYPES = {
    "dataset_overview",
    "analysis_recommendation",
    "analysis_planning",
}

MANIFEST_DRIVEN_GOAL_TYPES = {
    "dataset_overview",
    "analysis_recommendation",
    "analysis_planning",
    "regression_modeling",
    "visualization",
    "data_cleaning",
}


def _make_step_id(tool_name: str) -> str:
    safe_name = tool_name.replace(" ", "_").replace("-", "_")
    return f"step_{safe_name}_{uuid.uuid4().hex[:6]}"


def _capability_index(capability_map: CapabilityMap) -> Dict[str, AnalysisCapability]:
    return {
        capability.tool_name: capability
        for capability in capability_map.capabilities
    }


def _capability_to_step(
    capability: AnalysisCapability,
    *,
    purpose: str,
    variables: Optional[Dict[str, object]] = None,
    arguments: Optional[Dict[str, object]] = None,
    required_user_choices: Optional[List[str]] = None,
) -> PlanStep:
    variables = variables or {}
    arguments = arguments or {}
    manifest = _manifest_for_tool(capability.tool_name)

    requires_confirmation = (
        manifest.requires_confirmation
        if manifest is not None
        else capability.requires_confirmation
    )
    mutates_data = (
        manifest.mutates_data
        if manifest is not None
        else capability.mutates_data
    )

    execution_ready = (
        capability.status == "ready"
        and not capability.required_user_choices
        and not required_user_choices
        and not requires_confirmation
    )

    return PlanStep(
        step_id=_make_step_id(capability.tool_name),
        title=capability.display_name,
        tool_name=capability.tool_name,
        method_family=capability.method_family,
        status=capability.status,
        execution_ready=execution_ready,
        purpose=purpose,
        rationale=purpose,
        variables=variables,
        arguments=arguments,
        candidate_variables=capability.candidate_variables,
        required_user_choices=required_user_choices or capability.required_user_choices,
        applicability_check={
            "status": capability.status,
            "reason": capability.reason,
        },
        warnings=capability.warnings,
        suggested_alternatives=capability.suggested_alternatives,
        requires_confirmation=requires_confirmation,
        mutates_data=mutates_data,
    )


def _select_capabilities(
    capability_map: CapabilityMap,
    tool_names: Iterable[str],
    *,
    include_not_applicable: bool = False,
) -> tuple[List[AnalysisCapability], List[AnalysisCapability]]:
    index = _capability_index(capability_map)
    selected = []
    blocked = []

    for tool_name in tool_names:
        capability = index.get(tool_name)

        if capability is None:
            continue

        if capability.status in {"ready", "needs_user_choice"}:
            selected.append(capability)
        elif include_not_applicable:
            blocked.append(capability)

    return selected, blocked


def _numeric_column_count(profile: DatasetProfileV2) -> int:
    return sum(
        1
        for col in profile.columns.values()
        if col.semantic_type in {"continuous_numeric", "discrete_numeric"}
    )


def _step_variables_for_task(tool_name: str, task_spec: TaskSpec) -> Dict[str, object]:

    manifest_variables = _manifest_bound_arguments_for_task(tool_name, task_spec)

    if manifest_variables:
        return manifest_variables

    if tool_name == "run_multiple_regression":
        variables = {}
        if task_spec.target_variables:
            variables["target_col"] = task_spec.target_variables[0]
        if task_spec.predictor_variables:
            variables["feature_cols"] = task_spec.predictor_variables
        return variables

    if tool_name in {"regression_diagnostics", "generate_residual_histogram"}:
        variables = {}
        if task_spec.target_variables:
            variables["target_col"] = task_spec.target_variables[0]
        if task_spec.predictor_variables:
            variables["feature_cols"] = task_spec.predictor_variables
        return variables

    if tool_name == "clean_data":
        variables = {}
        if task_spec.target_variables:
            variables["columns"] = task_spec.target_variables
        return variables

    if tool_name in {"run_correlation_test", "generate_scatterplot"}:
        if len(task_spec.predictor_variables) >= 2:
            return {
                "x_col": task_spec.predictor_variables[0],
                "y_col": task_spec.predictor_variables[1],
            }

    return {}


def _step_arguments_for_task(tool_name: str, task_spec: TaskSpec) -> Dict[str, object]:
    variables = _step_variables_for_task(tool_name, task_spec)

    if tool_name == "generate_scatterplot":
        return {
            "x_column": variables.get("x_col"),
            "y_column": variables.get("y_col"),
        }

    return dict(variables)


def _required_choices_for_task(
    capability: AnalysisCapability,
    task_spec: TaskSpec,
) -> List[str]:
    missing = []
    tool_name = capability.tool_name

    if _tool_has_manifest_planning_bindings(tool_name):
        missing.extend(
            _required_choices_from_manifest(tool_name, task_spec)
        )
    else:
        if tool_name == "run_multiple_regression":
            if not task_spec.target_variables:
                missing.append("target_col")
            if not task_spec.predictor_variables:
                missing.append("feature_cols")

        if tool_name in {"regression_diagnostics", "generate_residual_histogram"}:
            if not task_spec.target_variables:
                missing.append("target_col")
            if not task_spec.predictor_variables:
                missing.append("feature_cols")

        if tool_name == "clean_data":
            for choice in ("action_type", "strategy"):
                if choice not in missing:
                    missing.append(choice)
            if not task_spec.target_variables:
                missing.append("columns")

    for choice in capability.required_user_choices:
        if choice not in missing:
            missing.append(choice)

    return missing

def _purpose_for_tool(tool_name: str, goal_type: str) -> str:
    purposes = {
        "inspect_dataset": "Inspect dataset shape, columns, and basic schema.",
        "missingness_report": "Assess missing values before recommending analyses.",
        "get_summary_stats": "Summarize numeric variables with descriptive statistics.",
        "summarize_columns": "Summarize column-level distributions and types.",
        "get_correlation_matrix": "Screen numeric associations only when enough numeric variables exist.",
        "run_multiple_regression": "Fit the requested regression model.",
        "regression_diagnostics": "Check model diagnostics after the regression fit.",
        "generate_residual_histogram": "Generate residual distribution evidence after fitting the model.",
        "run_correlation_test": "Test association between selected variables.",
        "generate_scatterplot": "Visualize the selected variables.",
        "clean_data": "Prepare a data modification proposal that requires user confirmation.",
    }

    return purposes.get(tool_name, f"Support the {goal_type} goal.")


def _manifest_for_tool(tool_name: str):
    try:
        plugin = get_plugin(tool_name)

        if plugin is None:
            return None

        return build_tool_manifest(plugin)
    except Exception:
        return None

def _binding_value_from_task_spec(
    task_spec: TaskSpec,
    binding: Dict[str, object],
) -> Optional[object]:
    task_field = binding.get("task_field")

    if not isinstance(task_field, str) or not task_field:
        return None

    value = getattr(task_spec, task_field, None)

    if value is None:
        return None

    if "index" in binding:
        index = binding.get("index")

        if not isinstance(index, int):
            return None

        if not isinstance(value, list):
            return None

        if index < 0 or index >= len(value):
            return None

        return value[index]

    if value == "" or value == []:
        return None

    return value


def _manifest_bound_arguments_for_task(
    tool_name: str,
    task_spec: TaskSpec,
) -> Dict[str, object]:
    manifest = _manifest_for_tool(tool_name)

    if manifest is None or not manifest.task_argument_bindings:
        return {}

    arguments = {}

    for binding in manifest.task_argument_bindings:
        argument = binding.get("argument")

        if not isinstance(argument, str) or not argument:
            continue

        value = _binding_value_from_task_spec(task_spec, binding)

        if value is None or value == "" or value == []:
            continue

        arguments[argument] = value

    return arguments


def _tool_has_manifest_planning_bindings(tool_name: str) -> bool:
    manifest = _manifest_for_tool(tool_name)

    if manifest is None:
        return False

    return bool(
        manifest.task_argument_bindings
        or manifest.required_planning_choices
    )


def _required_choices_from_manifest(
    tool_name: str,
    task_spec: TaskSpec,
) -> List[str]:
    manifest = _manifest_for_tool(tool_name)

    if manifest is None:
        return []

    missing = []

    for choice in manifest.required_planning_choices:
        if isinstance(choice, str) and choice not in missing:
            missing.append(choice)

    for binding in manifest.task_argument_bindings:
        required_choice = binding.get("required_choice")

        if not isinstance(required_choice, str) or not required_choice:
            continue

        value = _binding_value_from_task_spec(task_spec, binding)

        if value is None or value == "" or value == []:
            if required_choice not in missing:
                missing.append(required_choice)

    return missing

def _plan_purpose_for_tool(tool_name: str, goal_type: str) -> str:
    manifest = _manifest_for_tool(tool_name)

    if manifest and manifest.default_plan_purpose:
        return manifest.default_plan_purpose

    return _purpose_for_tool(tool_name, goal_type)

def _tools_for_goal(task_spec: TaskSpec, profile: DatasetProfileV2) -> List[str]:
    goal_type = task_spec.goal_type

    if goal_type in MANIFEST_DRIVEN_GOAL_TYPES:
        manifest_tools = _tools_for_goal_from_manifest(goal_type)

        if _manifest_selection_is_complete(
            goal_type=goal_type,
            manifest_tools=manifest_tools,
        ):
            tools = list(manifest_tools)
        else:
            tools = _legacy_tools_for_goal(goal_type)

        if goal_type in OVERVIEW_STYLE_GOAL_TYPES:
            if _numeric_column_count(profile) < 2 and "get_correlation_matrix" in tools:
                tools.remove("get_correlation_matrix")

        return _sort_selected_tools_by_manifest_order(tools)

    if goal_type == "association_analysis":
        return _sort_selected_tools_by_manifest_order(
            _legacy_tools_for_goal(goal_type)
        )

    return _sort_selected_tools_by_manifest_order(
        _legacy_tools_for_goal(goal_type)
    )


def _sort_selected_tools_by_manifest_order(tool_names: List[str]) -> List[str]:
    indexed = []

    for index, tool_name in enumerate(tool_names):
        manifest = _manifest_for_tool(tool_name)

        if manifest is None:
            return list(tool_names)

        plan_order = getattr(manifest, "plan_order", None)

        if type(plan_order) is not int:
            return list(tool_names)

        indexed.append((plan_order, index, tool_name))

    return [
        tool_name
        for _, _, tool_name in sorted(indexed)
    ]


def _tools_for_goal_from_manifest(goal_type: str) -> List[str]:
    try:
        selected = []

        for _, plugin in sorted(PLUGIN_REGISTRY.items()):
            manifest = build_tool_manifest(plugin)

            if goal_type not in manifest.supported_goal_types:
                continue

            plan_order = getattr(manifest, "plan_order", None)

            if type(plan_order) is not int:
                return []

            selected.append((plan_order, manifest.tool_name))

        return [
            tool_name
            for _, tool_name in sorted(selected)
        ]
    except Exception:
        return []

def _legacy_tools_for_goal(goal_type: str) -> List[str]:
    if goal_type in OVERVIEW_STYLE_GOAL_TYPES:
        return list(DATASET_OVERVIEW_TOOLS)

    if goal_type == "regression_modeling":
        return list(REGRESSION_TOOLS)

    if goal_type == "association_analysis":
        return list(ASSOCIATION_TOOLS)

    if goal_type == "visualization":
        return list(VISUALIZATION_TOOLS)

    if goal_type == "data_cleaning":
        return list(DATA_CLEANING_TOOLS)

    if goal_type == "eda":
        return list(DATASET_OVERVIEW_TOOLS[:3])

    return list(DATASET_OVERVIEW_TOOLS[:3])


def _manifest_selection_is_complete(
    *,
    goal_type: str,
    manifest_tools: List[str],
) -> bool:
    if not manifest_tools:
        return False

    legacy_tools = _legacy_tools_for_goal(goal_type)

    if goal_type not in MANIFEST_DRIVEN_GOAL_TYPES:
        return False

    return set(legacy_tools).issubset(set(manifest_tools))


def _tools_for_goal(task_spec: TaskSpec, profile: DatasetProfileV2) -> List[str]:
    goal_type = task_spec.goal_type

    if goal_type in {"dataset_overview", "analysis_recommendation", "analysis_planning"}:
        manifest_tools = _tools_for_goal_from_manifest(goal_type)

        if manifest_tools == DATASET_OVERVIEW_TOOLS:
            tools = list(manifest_tools)
        else:
            tools = list(DATASET_OVERVIEW_TOOLS)

        if _numeric_column_count(profile) < 2 and "get_correlation_matrix" in tools:
            tools.remove("get_correlation_matrix")
        return _sort_selected_tools_by_manifest_order(tools)

    if goal_type == "regression_modeling":
        manifest_tools = _tools_for_goal_from_manifest("regression_modeling")

        if manifest_tools == REGRESSION_TOOLS:
            return manifest_tools

        return _sort_selected_tools_by_manifest_order(list(REGRESSION_TOOLS))

    if goal_type == "association_analysis":
        return _sort_selected_tools_by_manifest_order(list(ASSOCIATION_TOOLS))

    if goal_type == "visualization":
        manifest_tools = _tools_for_goal_from_manifest("visualization")

        if manifest_tools:
            return manifest_tools

        return _sort_selected_tools_by_manifest_order(list(VISUALIZATION_TOOLS))

    if goal_type == "data_cleaning":
        manifest_tools = _tools_for_goal_from_manifest("data_cleaning")

        if manifest_tools:
            return manifest_tools

        return _sort_selected_tools_by_manifest_order(list(DATA_CLEANING_TOOLS))

    return _sort_selected_tools_by_manifest_order(DATASET_OVERVIEW_TOOLS[:3])


def _task_spec_from_state_or_request(state: dict, user_request: str) -> TaskSpec:
    raw_task_spec = state.get("task_spec")

    if isinstance(raw_task_spec, dict):
        return TaskSpec.model_validate(raw_task_spec)

    decision = decide_interaction_intent(user_request, state=state)
    if decision.task_spec is not None:
        return decision.task_spec

    return TaskSpec(
        goal_type="analysis_recommendation",
        user_goal="Recommend a safe initial analysis plan.",
        source_user_request=user_request,
        confidence=0.4,
    )


def create_plan_from_state(state: dict) -> PlanProposal:
    capability_map = CapabilityMap.model_validate(state.get("capability_map"))
    profile = DatasetProfileV2.model_validate(state.get("dataset_profile_v2"))
    user_request = state.get("user_request", "")
    task_spec = _task_spec_from_state_or_request(state, user_request)

    return create_plan(
        user_request=user_request,
        task_spec=task_spec,
        dataset_profile=profile,
        capability_map=capability_map,
    )


def create_plan(
    *,
    user_request: str,
    task_spec: TaskSpec,
    dataset_profile: DatasetProfileV2,
    capability_map: CapabilityMap,
) -> PlanProposal:
    tool_names = _tools_for_goal(task_spec, dataset_profile)
    selected, blocked_caps = _select_capabilities(
        capability_map,
        tool_names,
        include_not_applicable=True,
    )

    steps = []
    blocked_steps = []

    for capability in selected:
        variables = _step_variables_for_task(capability.tool_name, task_spec)
        arguments = _step_arguments_for_task(capability.tool_name, task_spec)
        required_choices = _required_choices_for_task(capability, task_spec)

        steps.append(
            _capability_to_step(
                capability,
                purpose=_plan_purpose_for_tool(
                    capability.tool_name,
                    task_spec.goal_type,
                ),
                variables=variables,
                arguments={
                    key: value
                    for key, value in arguments.items()
                    if value is not None and value != "" and value != []
                },
                required_user_choices=required_choices,
            )
        )

    for capability in blocked_caps:
        if task_spec.goal_type in {"dataset_overview", "analysis_recommendation", "analysis_planning"}:
            if capability.tool_name in DISALLOWED_FOR_OVERVIEW:
                continue
        blocked_steps.append(
            _capability_to_step(
                capability,
                purpose=_plan_purpose_for_tool(
                    capability.tool_name,
                    task_spec.goal_type,
                ),
            )
        )

    plan = PlanProposal(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        user_request=user_request,
        data_version_id=capability_map.data_version_id,
        mode="plan_only",
        status="draft",
        summary=(
            f"Generated a goal-driven plan for `{task_spec.goal_type}`. "
            "No tools have been executed."
        ),
        assumptions=[
            "This plan is based on the current TaskSpec, dataset profile, and plugin capability contracts.",
            "Steps that require user choices are not execution-ready.",
            "No analysis tools have been executed.",
        ],
        steps=steps,
        blocked_or_not_recommended=blocked_steps,
        requires_user_confirmation_before_execution=True,
        warnings=capability_map.warnings,
    )

    return verify_plan(plan, dataset_profile)
