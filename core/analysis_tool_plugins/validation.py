from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.analysis_tool_plugins import get_plugin
from core.schema import VerificationResult


def _type_ok(value: Any, expected_type: type) -> bool:
    if expected_type is object:
        return True

    if expected_type is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type is bool:
        return isinstance(value, bool)

    if expected_type is str:
        return isinstance(value, str)

    if expected_type is list:
        return isinstance(value, list)

    return isinstance(value, expected_type)


def _type_name(t: type) -> str:
    return getattr(t, "__name__", str(t))


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _profile_column_names(profile) -> List[str]:
    if profile is None:
        return []

    cols = getattr(profile, "columns", None)

    if isinstance(cols, dict):
        return list(cols.keys())

    if isinstance(profile, dict):
        cols = profile.get("columns", {})
        if isinstance(cols, dict):
            return list(cols.keys())

    return []


def _validate_required_and_types(
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    schema,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    missing_required = []
    wrong_types = []

    required = schema.required or {}
    optional = schema.optional or {}

    for arg_name, expected_type in required.items():
        if arg_name not in arguments or arguments[arg_name] is None:
            missing_required.append(arg_name)
            continue

        if not _type_ok(arguments[arg_name], expected_type):
            wrong_types.append({
                "argument": arg_name,
                "expected": _type_name(expected_type),
                "actual": type(arguments[arg_name]).__name__,
                "value_preview": repr(arguments[arg_name])[:200],
            })

    for arg_name, value in arguments.items():
        if arg_name in required:
            continue

        expected_type = optional.get(arg_name)

        # Unknown argument is not automatically rejected yet.
        # You may tighten this later.
        if expected_type is None:
            continue

        if value is not None and not _type_ok(value, expected_type):
            wrong_types.append({
                "argument": arg_name,
                "expected": _type_name(expected_type),
                "actual": type(value).__name__,
                "value_preview": repr(value)[:200],
            })

    return missing_required, wrong_types


def _validate_allowed_values(
    *,
    arguments: Dict[str, Any],
    schema,
) -> List[Dict[str, Any]]:
    violations = []

    for arg_name, allowed in (schema.allowed_values or {}).items():
        if arg_name not in arguments or arguments[arg_name] is None:
            continue

        actual = _normalize_scalar(arguments[arg_name])
        allowed_norm = [_normalize_scalar(x) for x in allowed]

        if actual not in allowed_norm:
            violations.append({
                "argument": arg_name,
                "actual": arguments[arg_name],
                "allowed": allowed,
            })

    return violations


def _validate_conditional_allowed_values(
    *,
    arguments: Dict[str, Any],
    schema,
) -> List[Dict[str, Any]]:
    violations = []

    rules = schema.conditional_allowed_values or {}

    for condition_arg, branches in rules.items():
        if condition_arg not in arguments:
            continue

        condition_value = _normalize_scalar(arguments.get(condition_arg))

        if condition_value not in branches:
            continue

        dependent_rules = branches[condition_value]

        for dependent_arg, allowed in dependent_rules.items():
            actual = _normalize_scalar(arguments.get(dependent_arg))
            allowed_norm = [_normalize_scalar(x) for x in allowed]

            if actual not in allowed_norm:
                violations.append({
                    "condition": {
                        "argument": condition_arg,
                        "value": condition_value,
                    },
                    "argument": dependent_arg,
                    "actual": arguments.get(dependent_arg),
                    "allowed": allowed,
                })

    return violations


def _validate_column_references(
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    schema,
    available_columns: List[str],
) -> List[Dict[str, Any]]:
    missing_columns = []

    for arg_name in schema.column_args or []:
        col = arguments.get(arg_name)

        if isinstance(col, str) and col not in available_columns:
            missing_columns.append({
                "argument": arg_name,
                "column": col,
            })

    for arg_name in schema.column_list_args or []:
        cols = arguments.get(arg_name)

        if cols == "all" and schema.allow_all_columns:
            continue

        if cols is None:
            continue

        if isinstance(cols, str):
            cols = [cols]

        if isinstance(cols, list):
            for col in cols:
                if isinstance(col, str) and col not in available_columns:
                    missing_columns.append({
                        "argument": arg_name,
                        "column": col,
                    })

    return missing_columns


def validate_plugin_action(action, profile=None) -> VerificationResult:
    """
    Canonical plugin-contract validator.

    This is the only runtime validator that should be used going forward.
    It validates against core.analysis_tool_plugins only.
    """
    tool_name = getattr(action, "tool_name", None)

    if not tool_name:
        return VerificationResult(
            action_id=getattr(action, "action_id", "unknown"),
            status="rejected_recoverable",
            feedback="Error: tool_name is missing.",
            error_code="MISSING_TOOL_NAME",
            details={},
        )

    plugin = get_plugin(tool_name)

    if plugin is None:
        return VerificationResult(
            action_id=getattr(action, "action_id", "unknown"),
            status="rejected_terminal",
            feedback=f"Error: tool '{tool_name}' is not registered in analysis_tool_plugins.",
            error_code="TOOL_NOT_REGISTERED",
            details={"tool_name": tool_name},
        )

    schema = plugin.argument_schema
    raw_arguments = getattr(action, "arguments", {}) or {}

    if not isinstance(raw_arguments, dict):
        return VerificationResult(
            action_id=getattr(action, "action_id", "unknown"),
            status="rejected_recoverable",
            feedback=f"Arguments for tool '{tool_name}' must be a dictionary.",
            error_code="ARGUMENTS_NOT_DICT",
            details={
                "tool_name": tool_name,
                "arguments_type": type(raw_arguments).__name__,
            },
        )

    canonical_arguments = schema.canonicalize_arguments(raw_arguments)

    missing_required, wrong_types = _validate_required_and_types(
        tool_name=tool_name,
        arguments=canonical_arguments,
        schema=schema,
    )

    invalid_values = _validate_allowed_values(
        arguments=canonical_arguments,
        schema=schema,
    )

    conditional_violations = _validate_conditional_allowed_values(
        arguments=canonical_arguments,
        schema=schema,
    )

    available_columns = _profile_column_names(profile)
    missing_columns = []

    if available_columns:
        missing_columns = _validate_column_references(
            tool_name=tool_name,
            arguments=canonical_arguments,
            schema=schema,
            available_columns=available_columns,
        )

    if missing_required or wrong_types or invalid_values or conditional_violations or missing_columns:
        return VerificationResult(
            action_id=getattr(action, "action_id", "unknown"),
            status="rejected_recoverable",
            feedback=(
                f"Tool argument validation failed for '{tool_name}'. "
                f"See details for required fields, types, allowed values, conditional rules, and columns."
            ),
            error_code="INVALID_TOOL_ARGUMENTS",
            details={
                "tool_name": tool_name,
                "missing_required_arguments": missing_required,
                "wrong_type_arguments": wrong_types,
                "invalid_values": invalid_values,
                "conditional_violations": conditional_violations,
                "missing_columns": missing_columns,
                "received_arguments": raw_arguments,
                "canonical_arguments": canonical_arguments,
            },
        )

    if canonical_arguments != raw_arguments:
        # Mutate current action deterministically after validation.
        # This is safe because canonicalization is plugin-owned and non-LLM.
        action.arguments = canonical_arguments

    if plugin.requires_confirmation:
        return VerificationResult(
            action_id=getattr(action, "action_id", "unknown"),
            status="needs_review",
            feedback=(
                f"Action '{tool_name}' mutates data or is high-risk; "
                f"user confirmation is required before execution."
            ),
            error_code=None,
            details={
                "tool_name": tool_name,
                "canonical_arguments": canonical_arguments,
                "requires_confirmation": True,
            },
        )

    return VerificationResult(
        action_id=getattr(action, "action_id", "unknown"),
        status="allowed",
        feedback="Validation passed; executing.",
        error_code=None,
        details={
            "tool_name": tool_name,
            "canonical_arguments": canonical_arguments,
            "requires_confirmation": False,
        },
    )