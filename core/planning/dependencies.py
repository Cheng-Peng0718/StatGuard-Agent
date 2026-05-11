from __future__ import annotations

from typing import Any, Iterable

from core.dataset_intelligence.schemas import DatasetProfileV2


DATA_CLEANING_TOOL = "clean_data"

MODELING_TOOLS = {
    "run_multiple_regression",
    "run_anova",
}


def _get_field(value: Any, field_name: str, default=None):
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def _set_field(value: Any, field_name: str, new_value: Any) -> None:
    if isinstance(value, dict):
        value[field_name] = new_value
    else:
        setattr(value, field_name, new_value)


def _step_tool_name(step: Any) -> str | None:
    return _get_field(step, "tool_name")


def _step_execution_status(step: Any) -> str | None:
    return _get_field(step, "execution_status")


def is_modeling_tool(tool_name: str | None) -> bool:
    return tool_name in MODELING_TOOLS


def profile_has_missing_values(profile: DatasetProfileV2) -> bool:
    columns = getattr(profile, "columns", {}) or {}

    for column in columns.values():
        n_missing = getattr(column, "n_missing", 0) or 0
        if n_missing > 0:
            return True

    return False


def has_clean_data_step(steps: Iterable[Any]) -> bool:
    return any(
        _step_tool_name(step) == DATA_CLEANING_TOOL
        for step in steps
    )


def clean_data_completed(steps: Iterable[Any]) -> bool:
    return any(
        _step_tool_name(step) == DATA_CLEANING_TOOL
        and _step_execution_status(step) == "completed"
        for step in steps
    )


def has_modeling_step(steps: Iterable[Any]) -> bool:
    return any(
        is_modeling_tool(_step_tool_name(step))
        for step in steps
    )


def should_require_cleaning_before_modeling(
    *,
    steps: Iterable[Any],
    profile: DatasetProfileV2,
) -> bool:
    steps = list(steps)

    return (
        profile_has_missing_values(profile)
        and has_modeling_step(steps)
        and has_clean_data_step(steps)
    )


def reorder_clean_data_before_modeling(plan: Any, profile: DatasetProfileV2) -> Any:
    """
    Ensure clean_data appears before the first modeling step when missingness exists.

    This only reorders plan steps. It does not make clean_data executable.
    clean_data still needs explicit choices and human review.
    """
    steps = list(getattr(plan, "steps", []) or [])

    if not should_require_cleaning_before_modeling(
        steps=steps,
        profile=profile,
    ):
        return plan

    first_model_idx = None
    clean_idx = None

    for idx, step in enumerate(steps):
        tool_name = _step_tool_name(step)

        if clean_idx is None and tool_name == DATA_CLEANING_TOOL:
            clean_idx = idx

        if first_model_idx is None and is_modeling_tool(tool_name):
            first_model_idx = idx

    if clean_idx is None or first_model_idx is None:
        return plan

    if clean_idx < first_model_idx:
        return plan

    clean_step = steps.pop(clean_idx)
    steps.insert(first_model_idx, clean_step)

    plan.steps = steps
    return plan


def modeling_blocked_by_pending_cleaning(
    *,
    step: Any,
    pending_plan: dict,
    profile: DatasetProfileV2,
) -> bool:
    """
    Runtime safety gate.

    Even if a modeling step becomes ready through user choices, it must not
    execute before clean_data completes when the dataset has missing values.
    """
    if not is_modeling_tool(_step_tool_name(step)):
        return False

    steps = pending_plan.get("steps", []) or []

    if not should_require_cleaning_before_modeling(
        steps=steps,
        profile=profile,
    ):
        return False

    return not clean_data_completed(steps)