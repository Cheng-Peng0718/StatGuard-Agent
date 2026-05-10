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
    compact_dict,
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


def execute_correlation_test(context) -> Dict[str, Any]:
    """
    Pairwise Pearson/Spearman correlation test.

    Args:
        x_col: first numeric column
        y_col: second numeric column
        method: 'pearson' or 'spearman', default 'pearson'
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        x_col = _get_arg(context, "x_col")
        y_col = _get_arg(context, "y_col")
        method = str(_get_arg(context, "method", "pearson")).lower().strip()

        if not x_col or not y_col:
            return _blocked(
                "MISSING_CORRELATION_ARGS",
                "x_col and y_col are required.",
                suggested_next_actions=[
                    "Specify two numeric columns for the correlation test."
                ],
            )

        missing_cols = [col for col in [x_col, y_col] if col not in df.columns]

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

        if method not in {"pearson", "spearman"}:
            return _blocked(
                "UNSUPPORTED_CORRELATION_METHOD",
                f"Unsupported correlation method: {method}",
                details={"method": method},
                suggested_next_actions=[
                    "Use method='pearson' or method='spearman'."
                ],
            )

        x = pd.to_numeric(df[x_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(df[y_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        work = pd.DataFrame({"x": x, "y": y}).dropna()
        nobs = int(len(work))

        if nobs < 3:
            return _blocked(
                "INSUFFICIENT_VALID_OBSERVATIONS",
                "Correlation test requires at least 3 complete numeric observations.",
                details={
                    "x_col": x_col,
                    "y_col": y_col,
                    "nobs": nobs,
                },
                suggested_next_actions=[
                    "Choose columns with more complete numeric observations."
                ],
            )

        if work["x"].nunique() <= 1 or work["y"].nunique() <= 1:
            return _blocked(
                "CONSTANT_INPUT",
                "Correlation is undefined when one selected column is constant.",
                details={
                    "x_col": x_col,
                    "y_col": y_col,
                    "x_unique": int(work["x"].nunique()),
                    "y_unique": int(work["y"].nunique()),
                },
                suggested_next_actions=[
                    "Choose two non-constant numeric columns."
                ],
            )

        if method == "pearson":
            correlation, p_value = stats.pearsonr(work["x"], work["y"])
        else:
            correlation, p_value = stats.spearmanr(work["x"], work["y"])

        return _ok(
            "Correlation test completed.",
            {
                "x_col": x_col,
                "y_col": y_col,
                "method": method,
                "nobs": nobs,
                "correlation": _round_or_none(correlation),
                "p_value": _round_or_none(p_value),
                "significant_at_0_05": (
                    bool(p_value < 0.05)
                    if math.isfinite(float(p_value))
                    else None
                ),
            },
        )

    except Exception as e:
        return _failed(
            "CORRELATION_TEST_EXCEPTION",
            "Correlation test failed.",
            e,
        )


def extract_correlation_test(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    x_col = payload.get("x_col") or arguments.get("x_col")
    y_col = payload.get("y_col") or arguments.get("y_col")
    method = payload.get("method")

    if x_col and y_col:
        title = f"Correlation Test: {x_col} vs {y_col}"
    else:
        title = "Correlation Test"

    metrics = compact_dict({
        "method": method,
        "nobs": payload.get("nobs"),
        "correlation": payload.get("correlation"),
        "p_value": payload.get("p_value"),
    })

    tables: Dict[str, Any] = {}

    metadata = compact_dict({
        "x_col": x_col,
        "y_col": y_col,
        "significant_at_0_05": payload.get("significant_at_0_05"),
    })

    summary = "Completed correlation test."

    if method:
        summary += f" Method: `{method}`."

    if x_col and y_col:
        summary += f" Variables: `{x_col}` and `{y_col}`."

    return title, summary, metrics, tables, metadata


CORRELATION_TEST_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "nobs": "Observations used",
            "correlation": "Correlation coefficient",
            "p_value": "p-value",
        },
        formatters={
            "correlation": lambda x: format_number(x, digits=4),
            "p_value": format_p_value,
        },
        order=[
            "method",
            "nobs",
            "correlation",
            "p_value",
        ],
    ),
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_correlation_test",
    display_name="Correlation Test",
    requires_confirmation=False,

    argument_schema=ArgumentSchema(
        required={
            "x_col": str,
            "y_col": str,
        },
        optional={
            "method": str,
        },
        column_args=[
            "x_col",
            "y_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
        allowed_values={
            "method": ["pearson", "spearman", "kendall"],
        },
        value_aliases={
            "method": {
                "pearson correlation": "pearson",
                "spearman correlation": "spearman",
                "spearman rank": "spearman",
                "kendall tau": "kendall",
                "kendall's tau": "kendall",
            },
        },
    ),

    execute=execute_correlation_test,
    extractor=extract_correlation_test,
    guardrail_evaluators=[],
    display_config=CORRELATION_TEST_DISPLAY,

    # Generic method/planning contract.
    method_family="association_test",

    # Correlation test requires two user-selected numeric variables.
    # It should appear in plans as needs_user_choice, not auto-ready.
    variable_roles=[
        VariableRoleSpec(
            role_name="x_col",
            required=True,
            user_must_select=True,
            allowed_semantic_types=[
                "continuous_numeric",
                "discrete_numeric",
            ],
            min_variables=1,
            max_variables=1,
            allow_auto_select=False,
            description=(
                "First numeric variable for the pairwise correlation test."
            ),
        ),
        VariableRoleSpec(
            role_name="y_col",
            required=True,
            user_must_select=True,
            allowed_semantic_types=[
                "continuous_numeric",
                "discrete_numeric",
            ],
            min_variables=1,
            max_variables=1,
            allow_auto_select=False,
            description=(
                "Second numeric variable for the pairwise correlation test."
            ),
        ),
    ],

    planning_policy=NEEDS_USER_VARIABLES_PLANNING,

    planning_metadata=PlanningMetadata(
        supported_goal_types=[
            "association_analysis",
        ],
        planning_tags=[
            "association",
            "correlation",
            "inferential",
        ],
        default_plan_purpose="Test association between selected variables.",
        expected_deliverables=[
            "association_test",
        ],
        task_argument_bindings=[
            {
                "task_field": "predictor_variables",
                "index": 0,
                "argument": "x_col",
                "required_choice": "x_col",
            },
            {
                "task_field": "predictor_variables",
                "index": 1,
                "argument": "y_col",
                "required_choice": "y_col",
            },
        ],
        plan_order=10,
    ),

    # Correlation test does not mutate data.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    repair_policy=DEFAULT_ANALYSIS_REPAIR,
))