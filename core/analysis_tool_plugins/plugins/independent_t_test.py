from typing import Any, Dict, Tuple
import math

import numpy as np
import pandas as pd
from scipy import stats

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    VariableRoleSpec,
    DisplayConfig,
    MetricDisplayConfig,
    compact_dict,
    format_bool_yes_no,
    format_number,
    format_p_value,
)
from core.analysis_tool_plugins.registry import register_plugin

from core.analysis_tool_plugins.policies import (
    NON_MUTATING_VERSIONING,
    DEFAULT_ANALYSIS_REPAIR,
    needs_user_choices,
)

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


def execute_independent_t_test(context) -> Dict[str, Any]:
    """
    Welch independent two-sample t-test.

    Args:
        target_col: numeric outcome
        group_col: grouping column
        group1_val: label for group 1
        group2_val: label for group 2
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
        group1_val = _get_arg(context, "group1_val")
        group2_val = _get_arg(context, "group2_val")

        if not target_col or not group_col or group1_val is None or group2_val is None:
            return _blocked(
                "MISSING_T_TEST_ARGS",
                "target_col, group_col, group1_val, and group2_val are required.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                    "group1_val": group1_val,
                    "group2_val": group2_val,
                },
                suggested_next_actions=[
                    "Specify a numeric target column, a grouping column, and two group values."
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

        y = pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        g1 = y[df[group_col] == group1_val].dropna()
        g2 = y[df[group_col] == group2_val].dropna()

        if len(g1) < 2 or len(g2) < 2:
            return _blocked(
                "INSUFFICIENT_GROUP_SIZE",
                "Each group needs at least 2 valid numeric observations.",
                details={
                    "group1": str(group1_val),
                    "group1_n": int(len(g1)),
                    "group2": str(group2_val),
                    "group2_n": int(len(g2)),
                },
                suggested_next_actions=[
                    "Choose groups with at least 2 valid numeric observations each."
                ],
            )

        t_stat, p_value = stats.ttest_ind(
            g1,
            g2,
            equal_var=False,
            nan_policy="omit",
        )

        details = {
            "method": "Welch two-sample t-test",
            "target_col": target_col,
            "group_col": group_col,
            "group1": str(group1_val),
            "group1_n": int(len(g1)),
            "group1_mean": _round_or_none(g1.mean()),
            "group1_std": _round_or_none(g1.std()),
            "group2": str(group2_val),
            "group2_n": int(len(g2)),
            "group2_mean": _round_or_none(g2.mean()),
            "group2_std": _round_or_none(g2.std()),
            "t_statistic": _round_or_none(t_stat),
            "p_value": _round_or_none(p_value),
            "significant_at_0_05": (
                bool(p_value < 0.05)
                if math.isfinite(float(p_value))
                else None
            ),
        }

        return _ok("Welch independent t-test completed.", details)

    except Exception as e:
        return _failed(
            "T_TEST_EXCEPTION",
            "T-test failed.",
            e,
        )


def extract_independent_t_test(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target_col = payload.get("target_col") or arguments.get("target_col")
    group_col = payload.get("group_col") or arguments.get("group_col")

    title = "Independent t-test"
    if target_col and group_col:
        title = f"Independent t-test: {target_col} by {group_col}"

    metrics = compact_dict({
        "method": payload.get("method"),
        "group1": payload.get("group1"),
        "group1_n": payload.get("group1_n"),
        "group1_mean": payload.get("group1_mean"),
        "group2": payload.get("group2"),
        "group2_n": payload.get("group2_n"),
        "group2_mean": payload.get("group2_mean"),
        "t_statistic": payload.get("t_statistic"),
        "p_value": payload.get("p_value"),
        "significant_at_0_05": payload.get("significant_at_0_05"),
    })

    tables: Dict[str, Any] = {}

    metadata = compact_dict({
        "target_col": target_col,
        "group_col": group_col,
        "group1_std": payload.get("group1_std"),
        "group2_std": payload.get("group2_std"),
    })

    summary = "Completed Welch independent two-sample t-test."
    if target_col and group_col:
        summary += f" Compared `{target_col}` across groups of `{group_col}`."

    return title, summary, metrics, tables, metadata


INDEPENDENT_T_TEST_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "group1": "Group 1",
            "group1_n": "Group 1 n",
            "group1_mean": "Group 1 mean",
            "group2": "Group 2",
            "group2_n": "Group 2 n",
            "group2_mean": "Group 2 mean",
            "t_statistic": "t statistic",
            "p_value": "p-value",
            "significant_at_0_05": "Significant at 0.05",
        },
        formatters={
            "group1_mean": lambda x: format_number(x, digits=4),
            "group2_mean": lambda x: format_number(x, digits=4),
            "t_statistic": lambda x: format_number(x, digits=4),
            "p_value": format_p_value,
            "significant_at_0_05": format_bool_yes_no,
        },
        order=[
            "method",
            "group1",
            "group1_n",
            "group1_mean",
            "group2",
            "group2_n",
            "group2_mean",
            "t_statistic",
            "p_value",
            "significant_at_0_05",
        ],
    ),
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_independent_t_test",
    display_name="Independent t-test",
    requires_confirmation=False,

    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "group_col": str,
            "group1_val": object,
            "group2_val": object,
        },
        optional={},
        column_args=[
            "target_col",
            "group_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),

    execute=execute_independent_t_test,
    extractor=extract_independent_t_test,
    guardrail_evaluators=[],
    display_config=INDEPENDENT_T_TEST_DISPLAY,

    # Generic method/planning contract.
    method_family="group_comparison",

    # Independent t-test requires:
    # - one continuous numeric target
    # - one grouping variable
    # - two specific group values selected by user or derived later by a checker
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
                "Continuous numeric outcome variable to compare between two groups."
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
                "Grouping variable defining the two independent groups."
            ),
        ),
    ],

    planning_policy=needs_user_choices(
        "target_col",
        "group_col",
        "group1_val",
        "group2_val",
    ),

    # Independent t-test does not mutate data.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    repair_policy=DEFAULT_ANALYSIS_REPAIR,
))