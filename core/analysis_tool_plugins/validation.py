from typing import Any, Dict, List, Optional

from core.analysis_tool_plugins import get_plugin


def _ok() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Tool call schema validation passed.",
        "recoverable": False,
        "details": {},
    }


def _blocked(
    *,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "error_code": "INVALID_TOOL_ARGUMENTS",
        "message": message,
        "recoverable": True,
        "details": details or {},
    }


def _warning(
    *,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": "warning",
        "message": message,
        "recoverable": False,
        "details": details or {},
    }


def _type_matches(value: Any, expected_type: type) -> bool:
    if expected_type is object:
        return True

    if expected_type is list:
        return isinstance(value, list)

    if expected_type is str:
        return isinstance(value, str)

    if expected_type is int:
        # Avoid bool passing as int.
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type is float:
        # int is acceptable where float is expected, but bool is not.
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if expected_type is bool:
        return isinstance(value, bool)

    return isinstance(value, expected_type)


def _get_profile_columns(profile: Any) -> List[str]:
    """
    Accepts several possible DatasetProfile shapes during migration.
    """
    if profile is None:
        return []

    # Pydantic/object style: profile.columns
    columns = getattr(profile, "columns", None)

    if isinstance(columns, list):
        if columns and isinstance(columns[0], dict):
            return [
                str(c.get("name"))
                for c in columns
                if isinstance(c, dict) and c.get("name") is not None
            ]

        return [str(c) for c in columns]

    # Dict style.
    if isinstance(profile, dict):
        columns = profile.get("columns")

        if isinstance(columns, list):
            if columns and isinstance(columns[0], dict):
                return [
                    str(c.get("name"))
                    for c in columns
                    if isinstance(c, dict) and c.get("name") is not None
                ]

            return [str(c) for c in columns]

    return []

def _validate_clean_data_semantics(arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate clean_data argument combinations beyond basic type checks.

    This prevents invalid mutating actions from reaching human approval.
    """
    action_type = str(arguments.get("action_type", "")).lower().strip()
    strategy = str(arguments.get("strategy", "")).lower().strip()

    if action_type == "drop":
        if strategy not in {"rows", "row", "drop_rows"}:
            return _blocked(
                message="For clean_data with action_type='drop', strategy must be 'rows'.",
                details={
                    "tool_name": "clean_data",
                    "action_type": action_type,
                    "strategy": strategy,
                    "allowed_strategies": ["rows"],
                },
            )

    elif action_type == "impute":
        if strategy not in {"mean", "median"}:
            return _blocked(
                message="For clean_data with action_type='impute', strategy must be 'mean' or 'median'.",
                details={
                    "tool_name": "clean_data",
                    "action_type": action_type,
                    "strategy": strategy,
                    "allowed_strategies": ["mean", "median"],
                },
            )

    else:
        return _blocked(
            message="For clean_data, action_type must be 'drop' or 'impute'.",
            details={
                "tool_name": "clean_data",
                "action_type": action_type,
                "allowed_action_types": ["drop", "impute"],
            },
        )

    return None


def validate_tool_call_schema(
    tool_name: str,
    arguments: Dict[str, Any] | None,
    profile: Any = None,
) -> Dict[str, Any]:
    """
    Unified tool-call schema validation.

    Final architecture:
    - Schemas live on AnalysisToolPlugin.argument_schema.
    - No dependency on tools/tool_schema.py.
    """
    arguments = arguments or {}

    plugin = get_plugin(tool_name)

    if plugin is None:
        return _warning(
            message=f"No unified plugin registered for tool {tool_name}; schema validation skipped.",
            details={"tool_name": tool_name},
        )

    schema = getattr(plugin, "argument_schema", None)

    if schema is None:
        return _warning(
            message=f"No argument schema registered for tool {tool_name}; schema validation skipped.",
            details={"tool_name": tool_name},
        )

    required = getattr(schema, "required", {}) or {}
    optional = getattr(schema, "optional", {}) or {}
    column_args = getattr(schema, "column_args", []) or []
    column_list_args = getattr(schema, "column_list_args", []) or []
    allow_all_columns = bool(getattr(schema, "allow_all_columns", False))

    missing_required = [
        name for name in required.keys()
        if name not in arguments or arguments.get(name) is None
    ]

    if missing_required:
        return _blocked(
            message=f"Missing required arguments for tool {tool_name}: {missing_required}",
            details={
                "tool_name": tool_name,
                "missing_required_arguments": missing_required,
            },
        )

    type_errors = []

    for name, expected_type in {**required, **optional}.items():
        if name not in arguments or arguments.get(name) is None:
            continue

        value = arguments.get(name)

        if not _type_matches(value, expected_type):
            type_errors.append({
                "argument": name,
                "expected_type": getattr(expected_type, "__name__", str(expected_type)),
                "actual_type": type(value).__name__,
                "value": value,
            })

    if type_errors:
        return _blocked(
            message=f"Invalid argument types for tool {tool_name}.",
            details={
                "tool_name": tool_name,
                "type_errors": type_errors,
            },
        )

    available_columns = _get_profile_columns(profile)

    # If no profile is available, skip column existence validation.
    if available_columns:
        missing_columns = []

        for arg_name in column_args:
            value = arguments.get(arg_name)

            if value is None:
                continue

            if value not in available_columns:
                missing_columns.append({
                    "argument": arg_name,
                    "column": value,
                })

        for arg_name in column_list_args:
            value = arguments.get(arg_name)

            if value is None:
                continue

            if value == "all" and allow_all_columns:
                continue

            if isinstance(value, str):
                # Some tools allow a single column string for a list-like arg.
                values = [value]
            elif isinstance(value, list):
                values = value
            else:
                continue

            for col in values:
                if col not in available_columns:
                    missing_columns.append({
                        "argument": arg_name,
                        "column": col,
                    })

        if missing_columns:
            return _blocked(
                message=f"Column validation failed for tool {tool_name}.",
                details={
                    "tool_name": tool_name,
                    "missing_columns": missing_columns,
                    "available_columns": available_columns,
                },
            )

    if tool_name == "clean_data":
        semantic_result = _validate_clean_data_semantics(arguments)
        if semantic_result is not None:
            return semantic_result

    return _ok()