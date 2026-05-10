from typing import Any, Dict, Tuple
import math

import numpy as np
import pandas as pd
from scipy import stats

from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.arguments import ArgumentSchema
from core.analysis_tool_plugins.roles import VariableRoleSpec
from core.analysis_tool_plugins.display import (
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_bool_yes_no,
    format_number,
    format_p_value,
)
from core.analysis_tool_plugins.registry import register_plugin

from core.analysis_tool_plugins.policies import (
    NEEDS_USER_VARIABLES_PLANNING,
    NON_MUTATING_VERSIONING,
    DEFAULT_ANALYSIS_REPAIR,
)
from core.analysis_tool_plugins.planning_contracts import PlanningMetadata

MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


def _ok(message: str, details: Dict[str, Any], artifacts=None):
    return {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }


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


def _round_or_none(x: Any, digits: int = 6):
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, digits)
    except Exception:
        return None


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


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_missing = {str(x).strip().lower() for x in MISSING_TOKENS}

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            def norm(x):
                if isinstance(x, str):
                    lx = x.strip().lower()
                    if lx in lower_missing:
                        return np.nan
                    return x.strip()
                return x

            df[col] = df[col].map(norm)

    return df.replace([np.inf, -np.inf], np.nan)


def execute_anova(context) -> Dict[str, Any]:
    """
    One-way ANOVA.

    Args:
        target_col: numeric outcome
        group_col: categorical grouping column
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        df = _standardize_dataframe(df)

        target_col = _get_arg(context, "target_col")
        group_col = _get_arg(context, "group_col")

        if not target_col or not group_col:
            return _blocked(
                "MISSING_ANOVA_ARGS",
                "target_col and group_col are required.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                },
                suggested_next_actions=[
                    "Specify a numeric target column and a categorical grouping column."
                ],
            )

        missing_cols = [c for c in [target_col, group_col] if c not in df.columns]

        if missing_cols:
            return _blocked(
                "COLUMNS_NOT_FOUND",
                f"Columns not found: {missing_cols}",
                details={
                    "missing_cols": missing_cols,
                    "available_columns": list(df.columns),
                },
                suggested_next_actions=[
                    "Inspect dataset columns and retry with valid column names."
                ],
            )

        work = pd.DataFrame({
            "y": pd.to_numeric(df[target_col], errors="coerce"),
            "g": df[group_col],
        }).replace([np.inf, -np.inf], np.nan).dropna()

        group_rows = []
        groups = []

        for group_value, values in work.groupby("g")["y"]:
            clean_values = values.dropna().astype(float)

            if len(clean_values) >= 2:
                groups.append(clean_values.to_numpy(dtype=float))

                group_rows.append({
                    "group": str(group_value),
                    "n": int(len(clean_values)),
                    "mean": _round_or_none(clean_values.mean()),
                    "std": _round_or_none(clean_values.std()),
                    "min": _round_or_none(clean_values.min()),
                    "max": _round_or_none(clean_values.max()),
                })

        if len(groups) < 2:
            return _blocked(
                "INSUFFICIENT_GROUPS",
                "ANOVA requires at least two groups with at least two valid numeric observations each.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                    "valid_group_count": int(len(groups)),
                    "group_summaries": group_rows,
                },
                suggested_next_actions=[
                    "Choose a grouping variable with at least two groups and enough numeric observations."
                ],
            )

        f_statistic, p_value = stats.f_oneway(*groups)

        details = {
            "method": "One-way ANOVA",
            "target_col": target_col,
            "group_col": group_col,
            "valid_group_count": int(len(groups)),
            "nobs": int(sum(len(g) for g in groups)),
            "F_statistic": _round_or_none(f_statistic),
            "p_value": _round_or_none(p_value),
            "significant_at_0_05": (
                bool(p_value < 0.05)
                if math.isfinite(float(p_value))
                else None
            ),
            "group_summaries": group_rows,
        }

        return _ok("One-way ANOVA completed.", details)

    except Exception as e:
        return _failed(
            "ANOVA_EXCEPTION",
            "ANOVA failed.",
            e,
        )


def extract_anova(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target_col = payload.get("target_col") or arguments.get("target_col")
    group_col = payload.get("group_col") or arguments.get("group_col")

    title = "One-way ANOVA"
    if target_col and group_col:
        title = f"One-way ANOVA: {target_col} by {group_col}"

    metrics = compact_dict({
        "method": payload.get("method"),
        "nobs": payload.get("nobs"),
        "valid_group_count": payload.get("valid_group_count"),
        "F_statistic": payload.get("F_statistic"),
        "p_value": payload.get("p_value"),
        "significant_at_0_05": payload.get("significant_at_0_05"),
    })

    tables: Dict[str, Any] = {}

    group_summaries = payload.get("group_summaries", [])
    if group_summaries:
        tables["group_summaries"] = group_summaries

    metadata = compact_dict({
        "target_col": target_col,
        "group_col": group_col,
    })

    summary = "Completed one-way ANOVA."
    if target_col and group_col:
        summary += f" Compared `{target_col}` across groups of `{group_col}`."

    return title, summary, metrics, tables, metadata


ANOVA_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "nobs": "Observations used",
            "valid_group_count": "Valid groups",
            "F_statistic": "F statistic",
            "p_value": "p-value",
            "significant_at_0_05": "Significant at 0.05",
        },
        formatters={
            "F_statistic": lambda x: format_number(x, digits=4),
            "p_value": format_p_value,
            "significant_at_0_05": format_bool_yes_no,
        },
        order=[
            "method",
            "nobs",
            "valid_group_count",
            "F_statistic",
            "p_value",
            "significant_at_0_05",
        ],
    ),
    tables={
        "group_summaries": TableDisplayConfig(
            column_labels={
                "group": "Group",
                "n": "n",
                "mean": "Mean",
                "std": "SD",
                "min": "Min",
                "max": "Max",
            },
            column_formatters={
                "mean": lambda x: format_number(x, digits=4),
                "std": lambda x: format_number(x, digits=4),
                "min": lambda x: format_number(x, digits=4),
                "max": lambda x: format_number(x, digits=4),
            },
            column_order=[
                "group",
                "n",
                "mean",
                "std",
                "min",
                "max",
            ],
        )
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_anova",
    display_name="One-way ANOVA",
    requires_confirmation=False,

    # Execution-time argument contract.
    # This is still used by validate_plugin_action before execution.
    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "group_col": str,
        },
        optional={},
        column_args=[
            "target_col",
            "group_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),

    execute=execute_anova,
    extractor=extract_anova,
    guardrail_evaluators=[],
    display_config=ANOVA_DISPLAY,

    # Phase 5A: generic method/planning contract.
    method_family="group_comparison",

    # Variable-role contract.
    # Use role names matching argument names.
    # This will make later PlanStep -> ActionProposal mapping cleaner.
    variable_roles=[
        VariableRoleSpec(
            role_name="target_col",
            required=True,
            user_must_select=True,
            allowed_semantic_types=[
                "continuous_numeric",
            ],
            min_variables=1,
            max_variables=1,
            allow_auto_select=False,
            description=(
                "Continuous numeric outcome variable to compare across groups."
            ),
        ),
        VariableRoleSpec(
            role_name="group_col",
            required=True,
            user_must_select=True,
            allowed_semantic_types=[
                "binary_categorical",
                "nominal_categorical",
                "ordinal_categorical",
                "discrete_numeric",
            ],
            min_variables=1,
            max_variables=1,
            allow_auto_select=False,
            description=(
                "Grouping variable. Numeric-coded grouping variables may appear "
                "as discrete_numeric and should be treated as factors."
            ),
        ),
    ],

    # Planning policy.
    # ANOVA requires user-selected variables, so it is NOT ready by default.
    planning_policy=NEEDS_USER_VARIABLES_PLANNING,

    planning_metadata=PlanningMetadata(
        supported_goal_types=[
            "group_comparison",
        ],
        not_recommended_for_goal_types=[
            "dataset_overview",
            "analysis_recommendation",
            "analysis_planning",
        ],
        planning_tags=[
            "group_comparison",
            "anova",
            "inferential",
        ],
        default_plan_purpose=(
            "Compare numeric outcomes across groups with one-way ANOVA."
        ),
        expected_deliverables=[
            "group_comparison_test",
        ],
        plan_order=10,
    ),

    # ANOVA does not mutate data.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    # Basic repair policy.
    repair_policy=DEFAULT_ANALYSIS_REPAIR,
))