from typing import Any, Dict

from core.analysis_tool_plugins.registry import get_plugin


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


def _profile_column_names(profile) -> list[str]:
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


def _schema_dict_for_plugin(tool_name: str) -> Dict[str, Any] | None:
    plugin = get_plugin(tool_name)

    if plugin is None:
        return None

    argument_schema = getattr(plugin, "argument_schema", None)

    if argument_schema is None:
        return None

    if hasattr(argument_schema, "to_legacy_schema_dict"):
        return argument_schema.to_legacy_schema_dict()

    return {
        "required": getattr(argument_schema, "required", {}) or {},
        "optional": getattr(argument_schema, "optional", {}) or {},
        "column_args": getattr(argument_schema, "column_args", []) or [],
        "column_list_args": getattr(argument_schema, "column_list_args", []) or [],
        "allow_all_columns": getattr(argument_schema, "allow_all_columns", False),
    }


def validate_tool_call_schema(
    tool_name: str,
    arguments: Dict[str, Any],
    profile=None,
) -> Dict[str, Any]:
    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return {
            "status": "blocked",
            "error_code": "ARGUMENTS_NOT_DICT",
            "message": f"Arguments for {tool_name} must be a dictionary.",
            "recoverable": True,
            "details": {
                "tool_name": tool_name,
                "arguments_type": type(arguments).__name__,
            },
        }

    schema = _schema_dict_for_plugin(tool_name)

    if schema is None:
        return {
            "status": "blocked",
            "error_code": "TOOL_NOT_REGISTERED",
            "message": f"Tool `{tool_name}` is not registered as an analysis tool plugin.",
            "recoverable": True,
            "details": {"tool_name": tool_name},
        }

    missing_required = []
    wrong_types = []

    for arg_name, expected_type in schema.get("required", {}).items():
        if arg_name not in arguments or arguments[arg_name] is None:
            missing_required.append(arg_name)
            continue

        if not _type_ok(arguments[arg_name], expected_type):
            wrong_types.append({
                "argument": arg_name,
                "expected": expected_type.__name__,
                "actual": type(arguments[arg_name]).__name__,
                "value_preview": repr(arguments[arg_name])[:200],
            })

    for arg_name, value in arguments.items():
        if arg_name in schema.get("required", {}):
            continue

        expected_type = schema.get("optional", {}).get(arg_name)

        if expected_type is not None and value is not None and not _type_ok(value, expected_type):
            wrong_types.append({
                "argument": arg_name,
                "expected": expected_type.__name__,
                "actual": type(value).__name__,
                "value_preview": repr(value)[:200],
            })

    if missing_required or wrong_types:
        return {
            "status": "blocked",
            "error_code": "INVALID_TOOL_ARGUMENTS",
            "message": f"Invalid arguments for tool {tool_name}.",
            "recoverable": True,
            "details": {
                "tool_name": tool_name,
                "missing_required_arguments": missing_required,
                "wrong_type_arguments": wrong_types,
                "received_arguments": arguments,
            },
        }

    available_columns = _profile_column_names(profile)

    if available_columns:
        col_check = _validate_column_references(
            tool_name,
            arguments,
            schema,
            available_columns,
        )

        if col_check["status"] != "ok":
            return col_check

    return {
        "status": "ok",
        "message": f"Tool call {tool_name} passed schema validation.",
        "recoverable": False,
        "details": {"tool_name": tool_name},
    }


def _validate_column_references(
    tool_name: str,
    arguments: Dict[str, Any],
    schema: Dict[str, Any],
    available_columns: list[str],
) -> Dict[str, Any]:
    missing_columns = []

    for arg_name in schema.get("column_args", []):
        col = arguments.get(arg_name)

        if isinstance(col, str) and col not in available_columns:
            missing_columns.append({
                "argument": arg_name,
                "column": col,
            })

    for arg_name in schema.get("column_list_args", []):
        cols = arguments.get(arg_name)

        if cols == "all" and schema.get("allow_all_columns", False):
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

    if missing_columns:
        return {
            "status": "blocked",
            "error_code": "COLUMN_NOT_FOUND",
            "message": "One or more referenced columns do not exist in the dataset.",
            "recoverable": True,
            "details": {
                "tool_name": tool_name,
                "missing_columns": missing_columns,
                "available_columns": available_columns,
            },
        }

    return {
        "status": "ok",
        "message": "Column validation passed.",
    }