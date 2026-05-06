from typing import Any, Dict, Tuple
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    VariableRoleSpec,
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


def _table_to_rows(table: pd.DataFrame) -> list[Dict[str, Any]]:
    rows = []

    for idx in table.index:
        row = {
            "row_level": str(idx),
        }

        for col in table.columns:
            value = table.loc[idx, col]
            row[str(col)] = int(value) if pd.notna(value) else None

        rows.append(row)

    return rows


def execute_chi_square(context) -> Dict[str, Any]:
    """
    Chi-square test of independence for two categorical variables.

    Args:
        row_col: first categorical variable
        col_col: second categorical variable
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        df = _standardize_dataframe(df)

        row_col = _get_arg(context, "row_col")
        col_col = _get_arg(context, "col_col")

        if not row_col or not col_col:
            return _blocked(
                "MISSING_CHI_SQUARE_ARGS",
                "row_col and col_col are required.",
                details={
                    "row_col": row_col,
                    "col_col": col_col,
                },
                suggested_next_actions=[
                    "Specify two categorical columns for the chi-square test."
                ],
            )

        missing_cols = [c for c in [row_col, col_col] if c not in df.columns]

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

        work = df[[row_col, col_col]].dropna()

        if len(work) == 0:
            return _blocked(
                "NO_COMPLETE_CASES",
                "No complete cases are available for the selected categorical columns.",
                details={
                    "row_col": row_col,
                    "col_col": col_col,
                },
                suggested_next_actions=[
                    "Choose columns with overlapping non-missing values."
                ],
            )

        contingency = pd.crosstab(work[row_col], work[col_col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return _blocked(
                "INSUFFICIENT_LEVELS",
                "Chi-square test requires at least 2 levels in each categorical variable.",
                details={
                    "row_col": row_col,
                    "col_col": col_col,
                    "table_shape": list(contingency.shape),
                },
                suggested_next_actions=[
                    "Choose two categorical columns with at least two observed levels each."
                ],
            )

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        expected_df = pd.DataFrame(
            expected,
            index=contingency.index,
            columns=contingency.columns,
        )

        expected_min = float(np.nanmin(expected))
        expected_lt_5 = int((expected < 5).sum())

        details = {
            "method": "Chi-square test of independence",
            "row_col": row_col,
            "col_col": col_col,
            "nobs": int(contingency.to_numpy().sum()),
            "row_levels": [str(x) for x in contingency.index.tolist()],
            "column_levels": [str(x) for x in contingency.columns.tolist()],
            "table_shape": list(contingency.shape),
            "chi_square_statistic": _round_or_none(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": _round_or_none(p_value),
            "significant_at_0_05": (
                bool(p_value < 0.05)
                if math.isfinite(float(p_value))
                else None
            ),
            "expected_min": _round_or_none(expected_min),
            "expected_cells_lt_5": expected_lt_5,
            "observed_table": _table_to_rows(contingency),
            "expected_table": _table_to_rows(expected_df.round(6)),
        }

        status = "ok"
        message = "Chi-square test completed."

        if expected_lt_5 > 0:
            status = "warning"
            message = (
                "Chi-square test completed, but some expected cell counts are below 5. "
                "Interpret the approximation cautiously."
            )

        return {
            "status": status,
            "message": message,
            "recoverable": False,
            "details": details,
            "artifacts": [],
        }

    except Exception as e:
        return _failed(
            "CHI_SQUARE_EXCEPTION",
            "Chi-square test failed.",
            e,
        )


def extract_chi_square(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    row_col = payload.get("row_col") or arguments.get("row_col")
    col_col = payload.get("col_col") or arguments.get("col_col")

    title = "Chi-square Test"
    if row_col and col_col:
        title = f"Chi-square Test: {row_col} × {col_col}"

    metrics = compact_dict({
        "method": payload.get("method"),
        "nobs": payload.get("nobs"),
        "chi_square_statistic": payload.get("chi_square_statistic"),
        "degrees_of_freedom": payload.get("degrees_of_freedom"),
        "p_value": payload.get("p_value"),
        "significant_at_0_05": payload.get("significant_at_0_05"),
        "expected_min": payload.get("expected_min"),
        "expected_cells_lt_5": payload.get("expected_cells_lt_5"),
    })

    tables: Dict[str, Any] = {}

    observed_table = payload.get("observed_table", [])
    if observed_table:
        tables["observed_table"] = observed_table

    # Expected table is useful, but more technical. Keep it in metadata by default.
    metadata = compact_dict({
        "row_col": row_col,
        "col_col": col_col,
        "row_levels": payload.get("row_levels"),
        "column_levels": payload.get("column_levels"),
        "table_shape": payload.get("table_shape"),
        "expected_table": payload.get("expected_table"),
    })

    summary = "Completed chi-square test of independence."
    if row_col and col_col:
        summary += f" Tested association between `{row_col}` and `{col_col}`."

    return title, summary, metrics, tables, metadata


CHI_SQUARE_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "nobs": "Observations used",
            "chi_square_statistic": "Chi-square statistic",
            "degrees_of_freedom": "Degrees of freedom",
            "p_value": "p-value",
            "significant_at_0_05": "Significant at 0.05",
            "expected_min": "Minimum expected count",
            "expected_cells_lt_5": "Expected cells below 5",
        },
        formatters={
            "chi_square_statistic": lambda x: format_number(x, digits=4),
            "p_value": format_p_value,
            "significant_at_0_05": format_bool_yes_no,
            "expected_min": lambda x: format_number(x, digits=4),
        },
        order=[
            "method",
            "nobs",
            "chi_square_statistic",
            "degrees_of_freedom",
            "p_value",
            "significant_at_0_05",
            "expected_min",
            "expected_cells_lt_5",
        ],
    ),
    tables={
        "observed_table": TableDisplayConfig(
            column_labels={
                "row_level": "Row level",
            },
            column_order=[
                "row_level",
            ],
        )
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_chi_square",
    display_name="Chi-square Test",
    requires_confirmation=False,

    # Execution-time argument contract.
    argument_schema=ArgumentSchema(
        required={
            "row_col": str,
            "col_col": str,
        },
        optional={},
        column_args=[
            "row_col",
            "col_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),

    execute=execute_chi_square,
    extractor=extract_chi_square,
    guardrail_evaluators=[],
    display_config=CHI_SQUARE_DISPLAY,

    # Generic method/planning contract.
    method_family="categorical_association",

    variable_roles=[
        VariableRoleSpec(
            role_name="row_col",
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
                "First categorical variable used as rows in the contingency table."
            ),
        ),
        VariableRoleSpec(
            role_name="col_col",
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
                "Second categorical variable used as columns in the contingency table."
            ),
        ),
    ],

    planning_policy=NEEDS_USER_VARIABLES_PLANNING,

    # Chi-square does not mutate data.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    repair_policy=DEFAULT_ANALYSIS_REPAIR,
))