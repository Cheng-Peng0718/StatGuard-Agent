from typing import Any, Dict, Tuple
import os
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.arguments import ArgumentSchema
from core.analysis_tool_plugins.roles import VariableRoleSpec
from core.analysis_tool_plugins.display import (
    DisplayConfig,
    MetricDisplayConfig,
    compact_dict,
    format_number,
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


def _workspace_dir(context) -> str:
    return getattr(context, "workspace_dir", ".") or "."


def _artifact_path(context, output_path: Any = None) -> str:
    if output_path:
        return str(output_path)

    artifact_dir = os.path.join(_workspace_dir(context), "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    return os.path.join(
        artifact_dir,
        f"scatterplot_{uuid.uuid4().hex[:8]}.png",
    )


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


def _plot_scatter(work: pd.DataFrame, x_col: str, y_col: str, path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(work[x_col], work[y_col], alpha=0.75)
    plt.title(f"Scatterplot: {y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def execute_scatterplot(context) -> Dict[str, Any]:
    """
    Generate a scatterplot for two numeric variables.

    Args:
        x_column: x-axis numeric column
        y_column: y-axis numeric column
        output_path: optional explicit output path
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        df = _standardize_dataframe(df)

        x_col = _get_arg(context, "x_column")
        y_col = _get_arg(context, "y_column")
        output_path = _get_arg(context, "output_path", None)

        if not x_col or not y_col:
            return _blocked(
                "MISSING_SCATTERPLOT_ARGS",
                "x_column and y_column are required.",
                details={
                    "x_column": x_col,
                    "y_column": y_col,
                },
                suggested_next_actions=[
                    "Specify x_column and y_column for the scatterplot."
                ],
            )

        missing_cols = [c for c in [x_col, y_col] if c not in df.columns]

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

        x = pd.to_numeric(df[x_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(df[y_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        work = pd.DataFrame({
            x_col: x,
            y_col: y,
        }).dropna()

        nobs = int(len(work))

        if nobs < 2:
            return _blocked(
                "INSUFFICIENT_VALID_OBSERVATIONS",
                "Scatterplot requires at least 2 complete numeric observations.",
                details={
                    "x_column": x_col,
                    "y_column": y_col,
                    "nobs": nobs,
                },
                suggested_next_actions=[
                    "Choose two numeric columns with at least 2 complete observations."
                ],
            )

        path = _artifact_path(context, output_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        _plot_scatter(work, x_col, y_col, path)

        details = {
            "x_column": x_col,
            "y_column": y_col,
            "nobs": nobs,
            "x_min": float(work[x_col].min()),
            "x_max": float(work[x_col].max()),
            "y_min": float(work[y_col].min()),
            "y_max": float(work[y_col].max()),
            "plot_path": path,
        }

        artifacts = [
            {
                "type": "png",
                "name": f"Scatterplot: {y_col} vs {x_col}",
                "path": path,
            }
        ]

        return _ok(
            "Scatterplot generated successfully.",
            details,
            artifacts=artifacts,
        )

    except Exception as e:
        return _failed(
            "SCATTERPLOT_EXCEPTION",
            "Scatterplot generation failed.",
            e,
        )


def extract_scatterplot(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    x_col = payload.get("x_column") or arguments.get("x_column")
    y_col = payload.get("y_column") or arguments.get("y_column")

    title = "Scatterplot"
    if x_col and y_col:
        title = f"Scatterplot: {y_col} vs {x_col}"

    metrics = compact_dict({
        "nobs": payload.get("nobs"),
        "x_min": payload.get("x_min"),
        "x_max": payload.get("x_max"),
        "y_min": payload.get("y_min"),
        "y_max": payload.get("y_max"),
    })

    tables: Dict[str, Any] = {}

    metadata = compact_dict({
        "x_column": x_col,
        "y_column": y_col,
        "plot_path": payload.get("plot_path"),
    })

    summary = "Generated a scatterplot."
    if x_col and y_col:
        summary += f" Plotted `{y_col}` against `{x_col}`."

    return title, summary, metrics, tables, metadata


SCATTERPLOT_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "nobs": "Observations plotted",
            "x_min": "X minimum",
            "x_max": "X maximum",
            "y_min": "Y minimum",
            "y_max": "Y maximum",
        },
        formatters={
            "x_min": lambda x: format_number(x, digits=4),
            "x_max": lambda x: format_number(x, digits=4),
            "y_min": lambda x: format_number(x, digits=4),
            "y_max": lambda x: format_number(x, digits=4),
        },
        order=[
            "nobs",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
        ],
    ),
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="generate_scatterplot",
    display_name="Scatterplot",
    requires_confirmation=False,

    argument_schema=ArgumentSchema(
        required={
            "x_column": str,
            "y_column": str,
        },
        optional={
            "output_path": str,
        },
        column_args=[
            "x_column",
            "y_column",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),

    execute=execute_scatterplot,
    extractor=extract_scatterplot,
    guardrail_evaluators=[],
    display_config=SCATTERPLOT_DISPLAY,

    # Generic method/planning contract.
    method_family="visualization",

    # Scatterplot requires two user-selected numeric variables.
    # It should appear in plans as needs_user_choice, not auto-ready.
    variable_roles=[
        VariableRoleSpec(
            role_name="x_column",
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
                "Numeric variable shown on the x-axis of the scatterplot."
            ),
        ),
        VariableRoleSpec(
            role_name="y_column",
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
                "Numeric variable shown on the y-axis of the scatterplot."
            ),
        ),
    ],

    planning_policy=NEEDS_USER_VARIABLES_PLANNING,

    planning_metadata=PlanningMetadata(
        supported_goal_types=[
            "association_analysis",
            "visualization",
        ],
        planning_tags=[
            "association",
            "visualization",
            "scatterplot",
        ],
        default_plan_purpose="Visualize the selected variables.",
        expected_deliverables=[
            "scatterplot",
        ],
        task_argument_bindings=[
            {
                "task_field": "predictor_variables",
                "index": 0,
                "argument": "x_column",
                "required_choice": "x_column",
            },
            {
                "task_field": "predictor_variables",
                "index": 1,
                "argument": "y_column",
                "required_choice": "y_column",
            },
        ],
        plan_order=10,
    ),

    # Scatterplot does not mutate the active dataset.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    repair_policy=DEFAULT_ANALYSIS_REPAIR,
))
