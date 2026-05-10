from typing import Any, Dict, Tuple
import uuid

import numpy as np
import pandas as pd

from core.data_versions import create_child_data_version, make_audit_event

from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.arguments import ArgumentSchema
from core.analysis_tool_plugins.roles import VariableRoleSpec
from core.analysis_tool_plugins.display import (
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin

from core.analysis_tool_plugins.policies import (
    MUTATING_CHILD_VERSIONING,
    DEFAULT_MUTATING_REPAIR,
    mutating_requires_choices,
)
from core.analysis_tool_plugins.planning_contracts import PlanningMetadata

def _ok(message: str, details: Dict[str, Any], artifacts=None, data_version_update=None):
    result = {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }

    # Put it both places during migration.
    # execution.py preserves top-level data_version_update into payload,
    # and details also carries it for downstream compatibility.
    if data_version_update is not None:
        result["data_version_update"] = data_version_update
        result["details"]["data_version_update"] = data_version_update

    return result


def _blocked(error_code: str, message: str, details=None, suggested_next_actions=None):
    result = {
        "status": "blocked",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": details or {},
        "artifacts": [],
    }

    if suggested_next_actions:
        result["suggested_next_actions"] = suggested_next_actions

    return result


def _failed(error_code: str, message: str, exc: Exception):
    return {
        "status": "failed",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "artifacts": [],
    }


def _get_arg(context, name: str, default: Any = None) -> Any:
    try:
        return context.get_arg(name, default)
    except TypeError:
        try:
            value = context.get_arg(name)
            return default if value is None else value
        except Exception:
            return default
    except Exception:
        return default


def _active_version_id(context) -> str:
    return (
        getattr(context, "active_data_version_id", None)
        or getattr(context, "current_data_version_id", None)
        or "unknown"
    )


def _make_version_id() -> str:
    return f"data_v_{uuid.uuid4().hex[:8]}"


def _normalize_columns_arg(columns):
    if columns is None:
        return []

    if isinstance(columns, str):
        return [columns]

    if isinstance(columns, list):
        return columns

    return []


def _selected_columns(df: pd.DataFrame, columns_arg):
    columns = _normalize_columns_arg(columns_arg)

    if not columns:
        return df.columns.tolist()

    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"Columns not found: {missing}")

    return columns


def _count_inf(df: pd.DataFrame) -> int:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return 0

    try:
        return int(np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum())
    except Exception:
        return 0


def _missing_by_columns(df: pd.DataFrame, columns: list[str]) -> Dict[str, int]:
    return {
        str(col): int(df[col].isna().sum())
        for col in columns
        if col in df.columns
    }


def _inf_by_columns(df: pd.DataFrame, columns: list[str]) -> Dict[str, int]:
    out = {}

    for col in columns:
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            out[str(col)] = 0
            continue

        try:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            out[str(col)] = int(np.isinf(arr).sum())
        except Exception:
            out[str(col)] = 0

    return out


def _impute_numeric_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].mean(skipna=True)
            df[col] = df[col].fillna(value)

    return df


def _impute_numeric_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median(skipna=True)
            df[col] = df[col].fillna(value)

    return df


def execute_clean_data(context) -> Dict[str, Any]:
    """
    Mutating data-cleaning tool.

    Supported actions:
        action_type='drop', strategy='rows'
        action_type='impute', strategy='mean'
        action_type='impute', strategy='median'

    Args:
        action_type: 'drop' or 'impute'
        strategy: 'rows', 'mean', or 'median'
        columns: optional list of columns. If omitted, all columns are considered.
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        action_type = str(_get_arg(context, "action_type", "")).lower().strip()
        strategy = str(_get_arg(context, "strategy", "")).lower().strip()
        columns_arg = _get_arg(context, "columns", None)

        if not action_type:
            return _blocked(
                "MISSING_CLEAN_ACTION",
                "action_type is required.",
                suggested_next_actions=[
                    "Use action_type='drop' or action_type='impute'."
                ],
            )

        try:
            cols = _selected_columns(df, columns_arg)
        except ValueError as e:
            return _blocked(
                "COLUMNS_NOT_FOUND",
                str(e),
                details={
                    "requested_columns": columns_arg,
                    "available_columns": list(df.columns),
                },
                suggested_next_actions=[
                    "Inspect dataset columns and retry with valid column names."
                ],
            )

        original_shape = tuple(df.shape)
        total_missing_before = int(df.isna().sum().sum())
        total_inf_before = _count_inf(df)
        selected_missing_before = _missing_by_columns(df, cols)
        selected_inf_before = _inf_by_columns(df, cols)

        # Treat inf as missing before cleaning.
        work = df.copy().replace([np.inf, -np.inf], np.nan)

        if action_type == "drop":
            if strategy not in {"rows", "row", "drop_rows"}:
                return _blocked(
                    "UNSUPPORTED_CLEAN_STRATEGY",
                    "For action_type='drop', only strategy='rows' is supported.",
                    details={
                        "action_type": action_type,
                        "strategy": strategy,
                    },
                )

            new_df = work.dropna(subset=cols).copy()
            normalized_strategy = "rows"

        elif action_type == "impute":
            if strategy == "mean":
                new_df = _impute_numeric_mean(work, cols)
            elif strategy == "median":
                new_df = _impute_numeric_median(work, cols)
            else:
                return _blocked(
                    "UNSUPPORTED_CLEAN_STRATEGY",
                    "For action_type='impute', supported strategies are 'mean' and 'median'.",
                    details={
                        "action_type": action_type,
                        "strategy": strategy,
                    },
                )

            normalized_strategy = strategy

        else:
            return _blocked(
                "UNSUPPORTED_CLEAN_ACTION",
                f"Unsupported action_type: {action_type}",
                details={"action_type": action_type},
                suggested_next_actions=[
                    "Use action_type='drop' or action_type='impute'."
                ],
            )

        new_df = new_df.reset_index(drop=True)

        final_shape = tuple(new_df.shape)
        total_missing_after = int(new_df.isna().sum().sum())
        total_inf_after = _count_inf(new_df)
        selected_missing_after = _missing_by_columns(new_df, cols)
        selected_inf_after = _inf_by_columns(new_df, cols)

        old_version_id = _active_version_id(context)

        new_version = create_child_data_version(
            df=new_df,
            workspace_dir=context.workspace_dir,
            parent_version_id=old_version_id,
            operation=f"clean_data:{action_type}-{normalized_strategy}",
            created_by="clean_data",
            description=(
                f"Cleaned data using action_type={action_type}, "
                f"strategy={normalized_strategy}, columns={cols}."
            ),
            metadata={
                "action_type": action_type,
                "strategy": normalized_strategy,
                "selected_columns": cols,
                "original_shape": original_shape,
                "final_shape": final_shape,
                "rows_removed": int(original_shape[0] - final_shape[0]),
                "total_missing_before": total_missing_before,
                "total_missing_after": total_missing_after,
            },
        )

        audit_event = make_audit_event(
            event_type="data_cleaned",
            version_id=new_version["version_id"],
            parent_version_id=old_version_id,
            tool_name="clean_data",
            description=(
                f"Created new data version {new_version['version_id']} "
                f"from {old_version_id}."
            ),
            details={
                "action_type": action_type,
                "strategy": normalized_strategy,
                "selected_columns": cols,
                "old_shape": tuple(original_shape),
                "new_shape": tuple(final_shape),
                "rows_removed": int(original_shape[0] - final_shape[0]),
            },
        )

        data_version_update = {
            "new_version": new_version,
            "active_data_version_id": new_version["version_id"],
            "audit_event": audit_event,
        }

        details = {
            "action_type": action_type,
            "strategy": normalized_strategy,
            "selected_columns": cols,
            "original_shape": str(original_shape),
            "final_shape": str(final_shape),
            "original_n_rows": int(original_shape[0]),
            "original_n_cols": int(original_shape[1]),
            "final_n_rows": int(final_shape[0]),
            "final_n_cols": int(final_shape[1]),
            "rows_removed": int(original_shape[0] - final_shape[0]),
            "total_missing_before": total_missing_before,
            "total_missing_after": total_missing_after,
            "total_inf_before": total_inf_before,
            "total_inf_after": total_inf_after,
            "selected_missing_before": selected_missing_before,
            "selected_missing_after": selected_missing_after,
            "selected_inf_before": selected_inf_before,
            "selected_inf_after": selected_inf_after,
            "final_columns": list(new_df.columns),
            "old_version_id": old_version_id,
            "new_version_id": new_version["version_id"],
            "data_version_created": True,
        }

        return _ok(
            "Data cleaning completed and a new data version was created.",
            details,
            data_version_update=data_version_update,
        )

    except Exception as e:
        return _failed(
            "CLEAN_DATA_EXCEPTION",
            "Data cleaning failed.",
            e,
        )


def extract_clean_data(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Data Cleaning"

    metrics = compact_dict({
        "original_n_rows": payload.get("original_n_rows"),
        "final_n_rows": payload.get("final_n_rows"),
        "rows_removed": payload.get("rows_removed"),
        "total_missing_before": payload.get("total_missing_before"),
        "total_missing_after": payload.get("total_missing_after"),
        "total_inf_before": payload.get("total_inf_before"),
        "total_inf_after": payload.get("total_inf_after"),
    })

    tables: Dict[str, Any] = {}

    selected_missing_after = payload.get("selected_missing_after", {})
    if isinstance(selected_missing_after, dict) and selected_missing_after:
        tables["selected_missing_after"] = [
            {
                "column": col,
                "missing_after": value,
            }
            for col, value in selected_missing_after.items()
        ]

    metadata = compact_dict({
        "action_type": payload.get("action_type"),
        "strategy": payload.get("strategy"),
        "selected_columns": payload.get("selected_columns"),
        "old_version_id": payload.get("old_version_id"),
        "new_version_id": payload.get("new_version_id"),
        "data_version_update": payload.get("data_version_update"),
        "final_columns": payload.get("final_columns"),
    })

    summary = "Cleaned the active dataset and created a new data version."

    if payload.get("action_type") and payload.get("strategy"):
        summary += f" Action: `{payload.get('action_type')}` using strategy `{payload.get('strategy')}`."

    if payload.get("new_version_id"):
        summary += f" New version: `{payload.get('new_version_id')}`."

    return title, summary, metrics, tables, metadata


CLEAN_DATA_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "original_n_rows": "Original rows",
            "final_n_rows": "Final rows",
            "rows_removed": "Rows removed",
            "total_missing_before": "Missing values before",
            "total_missing_after": "Missing values after",
            "total_inf_before": "Infinite values before",
            "total_inf_after": "Infinite values after",
        },
        formatters={
            "original_n_rows": lambda x: format_number(x, digits=0),
            "final_n_rows": lambda x: format_number(x, digits=0),
            "rows_removed": lambda x: format_number(x, digits=0),
            "total_missing_before": lambda x: format_number(x, digits=0),
            "total_missing_after": lambda x: format_number(x, digits=0),
            "total_inf_before": lambda x: format_number(x, digits=0),
            "total_inf_after": lambda x: format_number(x, digits=0),
        },
        order=[
            "original_n_rows",
            "final_n_rows",
            "rows_removed",
            "total_missing_before",
            "total_missing_after",
            "total_inf_before",
            "total_inf_after",
        ],
    ),
    tables={
        "selected_missing_after": TableDisplayConfig(
            column_labels={
                "column": "Column",
                "missing_after": "Missing after cleaning",
            },
            column_order=[
                "column",
                "missing_after",
            ],
        )
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="clean_data",
    display_name="Data Cleaning",
    requires_confirmation=True,

    argument_schema=ArgumentSchema(
        required={
            "action_type": str,
            "strategy": str,
        },
        optional={
            "columns": object,
        },
        column_args=[],
        column_list_args=["columns"],
        allow_all_columns=True,
        allowed_values={
            "action_type": ["drop", "impute"],
            "strategy": ["rows", "mean", "median"],
        },
        conditional_allowed_values={
            "action_type": {
                "drop": {
                    "strategy": ["rows"],
                },
                "impute": {
                    "strategy": ["mean", "median"],
                },
            },
        },
        value_aliases={
            "action_type": {
                "drop rows": "drop",
                "remove": "drop",
                "remove rows": "drop",
                "delete": "drop",
                "impute missing": "impute",
                "fill": "impute",
            },
            "strategy": {
                "row": "rows",
                "drop_rows": "rows",
                "drop rows": "rows",
                "drop": "rows",
                "average": "mean",
                "avg": "mean",
                "med": "median",
            },
        },
    ),

    execute=execute_clean_data,
    extractor=extract_clean_data,
    guardrail_evaluators=[],
    display_config=CLEAN_DATA_DISPLAY,

    # Generic method/planning contract.
    method_family="data_cleaning",

    # clean_data should not become execution-ready automatically from a generic plan.
    # It mutates data and requires explicit user intent / confirmation.
    variable_roles=[
        VariableRoleSpec(
            role_name="columns",
            required=False,
            user_must_select=True,
            allowed_semantic_types=[
                "continuous_numeric",
                "discrete_numeric",
                "binary_categorical",
                "nominal_categorical",
                "ordinal_categorical",
                "datetime",
                "text",
                "id_like",
                "unknown",
                "constant",
            ],
            min_variables=1,
            max_variables=None,
            allow_auto_select=False,
            description=(
                "Columns to clean. If omitted, the cleaning operation may apply "
                "to eligible columns according to the requested strategy."
            ),
        ),
    ],

    planning_policy=mutating_requires_choices(
        "action_type",
        "strategy",
    ),

    planning_metadata=PlanningMetadata(
        supported_goal_types=[
            "data_cleaning",
        ],
        not_recommended_for_goal_types=[
            "dataset_overview",
            "analysis_recommendation",
            "analysis_planning",
        ],
        planning_tags=[
            "data_cleaning",
            "mutation",
            "requires_confirmation",
        ],
        default_plan_purpose=(
            "Prepare a data modification proposal that requires user confirmation."
        ),
        expected_deliverables=[
            "cleaned_dataset_version",
        ],
        task_argument_bindings=[
            {
                "task_field": "target_variables",
                "argument": "columns",
                "required_choice": "columns",
            },
        ],
        required_planning_choices=[
            "action_type",
            "strategy",
        ],
        plan_order=10,
    ),

    # clean_data mutates data and must create a child data version.
    mutates_data=True,
    versioning_policy=MUTATING_CHILD_VERSIONING,

    repair_policy=DEFAULT_MUTATING_REPAIR,
))