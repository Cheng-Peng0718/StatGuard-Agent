from typing import Any, Dict, Tuple
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_bool_yes_no,
    format_number,
    format_p_value,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.guardrails import _new_finding


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


def _table_to_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    for idx in table.index:
        row = {"row_level": str(idx)}

        for col in table.columns:
            value = table.loc[idx, col]
            row[str(col)] = (
                int(value) if isinstance(value, (int, np.integer)) else
                float(value) if pd.notna(value) else None
            )

        rows.append(row)

    return rows


def _cramers_v(chi2: float, n: int, table_shape: tuple[int, int]) -> float | None:
    """
    Cramer's V effect size for chi-square test.

    V = sqrt( chi2 / (n * (min(rows, cols) - 1)) ),  range: [0, 1]

    Cohen's rules of thumb (depend on df = min(r,c)-1):
      df=1: small=0.10, medium=0.30, large=0.50
      df=2: small=0.07, medium=0.21, large=0.35
      df>=3: small=0.06, medium=0.17, large=0.29
    """
    rows, cols = table_shape
    k = min(rows, cols)

    if n <= 0 or k <= 1:
        return None

    try:
        v = math.sqrt(float(chi2) / (n * (k - 1)))
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _interpret_cramers_v(v: float | None, table_shape: tuple[int, int]) -> str | None:
    if v is None:
        return None

    rows, cols = table_shape
    df = min(rows, cols) - 1

    if df <= 0:
        return None

    if df == 1:
        small, medium, large = 0.10, 0.30, 0.50
    elif df == 2:
        small, medium, large = 0.07, 0.21, 0.35
    else:
        small, medium, large = 0.06, 0.17, 0.29

    a = abs(v)

    if a < small:
        return "negligible"
    if a < medium:
        return "small"
    if a < large:
        return "medium"

    return "large"


def execute_chi_square(context) -> Dict[str, Any]:
    """
    Chi-square test of independence for two categorical variables.

    For 2x2 tables with low expected counts (<5 in any cell), Fisher's exact
    test is reported as a small-sample alternative. Cramer's V effect size is
    always reported.

    Args:
        row_col: first categorical variable
        col_col: second categorical variable
        yates_correction: optional, default True for 2x2 tables only
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
        yates_correction = bool(_get_arg(context, "yates_correction", True))

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

        is_2x2 = contingency.shape == (2, 2)
        apply_yates = bool(yates_correction and is_2x2)

        # Primary chi-square test
        try:
            chi2, p_value, dof, expected = chi2_contingency(
                contingency,
                correction=apply_yates,
            )
        except Exception as e:
            return _failed(
                "CHI_SQUARE_COMPUTE_FAIL",
                "scipy chi2_contingency failed.",
                e,
            )

        expected_df = pd.DataFrame(
            expected,
            index=contingency.index,
            columns=contingency.columns,
        )

        expected_min = float(np.nanmin(expected))
        expected_lt_5 = int((expected < 5).sum())
        n_total = int(contingency.to_numpy().sum())

        # Effect size: Cramer's V (always)
        cramers_v_value = _cramers_v(chi2, n_total, contingency.shape)
        cramers_v_magnitude = _interpret_cramers_v(cramers_v_value, contingency.shape)

        # Fisher's exact for 2x2 with low expected counts
        fisher_exact_result: dict[str, Any] = {
            "applicable": False,
            "reason": "Fisher's exact is only computed for 2x2 tables with low expected counts.",
        }

        if is_2x2 and expected_lt_5 > 0:
            try:
                table_2x2 = contingency.to_numpy().astype(int)
                fe_odds_ratio, fe_p = fisher_exact(table_2x2, alternative="two-sided")

                fisher_exact_result = {
                    "applicable": True,
                    "method": "Fisher's exact test (two-sided)",
                    "odds_ratio": _round_or_none(fe_odds_ratio),
                    "p_value": _round_or_none(fe_p),
                    "significant_at_0_05": (
                        bool(fe_p < 0.05) if math.isfinite(float(fe_p)) else None
                    ),
                    "note": (
                        "Fisher's exact test is reported because at least one expected cell "
                        "count was below 5, which weakens the chi-square approximation."
                    ),
                }
            except Exception as e:
                fisher_exact_result = {
                    "applicable": True,
                    "method": "Fisher's exact test (two-sided)",
                    "error": str(e),
                    "note": "Fisher's exact failed to compute.",
                }

        method_label = "Chi-square test of independence"

        assumptions = [
            "Chi-square tests independence between two categorical variables.",
            "Observations are assumed to be independent.",
            "The chi-square approximation requires sufficiently large expected cell counts (commonly all >= 5).",
        ]

        if expected_lt_5 > 0:
            assumptions.append(
                f"{expected_lt_5} expected cell(s) are below 5; interpret the chi-square approximation cautiously."
            )

            if is_2x2:
                assumptions.append(
                    "Fisher's exact test is also reported and is preferred over chi-square for small 2x2 tables."
                )
            else:
                assumptions.append(
                    "Consider combining sparse categories or collecting more data."
                )

        details = {
            "method": method_label,
            "row_col": row_col,
            "col_col": col_col,
            "nobs": n_total,
            "row_levels": [str(x) for x in contingency.index.tolist()],
            "column_levels": [str(x) for x in contingency.columns.tolist()],
            "table_shape": list(contingency.shape),
            "yates_correction_applied": apply_yates,
            "chi_square_statistic": _round_or_none(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": _round_or_none(p_value),
            "significant_at_0_05": (
                bool(p_value < 0.05) if math.isfinite(float(p_value)) else None
            ),
            "expected_min": _round_or_none(expected_min),
            "expected_cells_lt_5": expected_lt_5,
            "cramers_v": _round_or_none(cramers_v_value),
            "cramers_v_magnitude": cramers_v_magnitude,
            "fisher_exact": fisher_exact_result,
            "observed_table": _table_to_rows(contingency),
            "expected_table": _table_to_rows(expected_df.round(6)),
            "assumptions_and_limitations": assumptions,
        }

        status = "ok"
        message = "Chi-square test completed."

        if expected_lt_5 > 0:
            status = "warning"
            if is_2x2:
                message = (
                    "Chi-square test completed, but some expected cell counts are below 5. "
                    "Fisher's exact test reported alongside as the small-sample alternative."
                )
            else:
                message = (
                    "Chi-square test completed, but some expected cell counts are below 5. "
                    "Interpret the chi-square approximation cautiously."
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

    fisher = payload.get("fisher_exact", {}) or {}

    metrics = compact_dict({
        "method": payload.get("method"),
        "nobs": payload.get("nobs"),
        "yates_correction_applied": payload.get("yates_correction_applied"),
        "chi_square_statistic": payload.get("chi_square_statistic"),
        "degrees_of_freedom": payload.get("degrees_of_freedom"),
        "p_value": payload.get("p_value"),
        "significant_at_0_05": payload.get("significant_at_0_05"),
        "cramers_v": payload.get("cramers_v"),
        "cramers_v_magnitude": payload.get("cramers_v_magnitude"),
        "expected_min": payload.get("expected_min"),
        "expected_cells_lt_5": payload.get("expected_cells_lt_5"),
        "fisher_exact_applicable": fisher.get("applicable"),
        "fisher_exact_p_value": fisher.get("p_value"),
        "fisher_exact_odds_ratio": fisher.get("odds_ratio"),
        "fisher_exact_significant_at_0_05": fisher.get("significant_at_0_05"),
    })

    tables: Dict[str, Any] = {}

    observed_table = payload.get("observed_table", [])
    if observed_table:
        tables["observed_table"] = observed_table

    if payload.get("assumptions_and_limitations"):
        tables["assumptions_and_limitations"] = [
            {"item": item} for item in payload.get("assumptions_and_limitations", [])
        ]

    metadata = compact_dict({
        "row_col": row_col,
        "col_col": col_col,
        "row_levels": payload.get("row_levels"),
        "column_levels": payload.get("column_levels"),
        "table_shape": payload.get("table_shape"),
        "expected_table": payload.get("expected_table"),
        "fisher_exact": payload.get("fisher_exact"),
    })

    summary = "Completed chi-square test of independence."
    if row_col and col_col:
        summary += f" Tested association between `{row_col}` and `{col_col}`."

    if payload.get("cramers_v") is not None:
        summary += (
            f" Cramer's V = {payload.get('cramers_v')} "
            f"({payload.get('cramers_v_magnitude') or 'magnitude n/a'})."
        )

    if fisher.get("applicable"):
        summary += f" Fisher's exact p-value: {fisher.get('p_value')}."

    return title, summary, metrics, tables, metadata


CHI_SQUARE_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "nobs": "Observations used",
            "yates_correction_applied": "Yates correction applied",
            "chi_square_statistic": "Chi-square statistic",
            "degrees_of_freedom": "Degrees of freedom",
            "p_value": "p-value",
            "significant_at_0_05": "Significant at 0.05",
            "cramers_v": "Cramer's V",
            "cramers_v_magnitude": "Cramer's V magnitude",
            "expected_min": "Minimum expected count",
            "expected_cells_lt_5": "Expected cells below 5",
            "fisher_exact_applicable": "Fisher's exact reported",
            "fisher_exact_p_value": "Fisher's exact p-value",
            "fisher_exact_odds_ratio": "Fisher's exact odds ratio",
            "fisher_exact_significant_at_0_05": "Fisher's exact significant at 0.05",
        },
        formatters={
            "chi_square_statistic": lambda x: format_number(x, digits=4),
            "p_value": format_p_value,
            "significant_at_0_05": format_bool_yes_no,
            "yates_correction_applied": format_bool_yes_no,
            "cramers_v": lambda x: format_number(x, digits=4),
            "expected_min": lambda x: format_number(x, digits=4),
            "fisher_exact_applicable": format_bool_yes_no,
            "fisher_exact_p_value": format_p_value,
            "fisher_exact_odds_ratio": lambda x: format_number(x, digits=4),
            "fisher_exact_significant_at_0_05": format_bool_yes_no,
        },
        order=[
            "method",
            "nobs",
            "yates_correction_applied",
            "chi_square_statistic",
            "degrees_of_freedom",
            "p_value",
            "significant_at_0_05",
            "cramers_v",
            "cramers_v_magnitude",
            "expected_min",
            "expected_cells_lt_5",
            "fisher_exact_applicable",
            "fisher_exact_p_value",
            "fisher_exact_odds_ratio",
            "fisher_exact_significant_at_0_05",
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
        ),
        "assumptions_and_limitations": TableDisplayConfig(
            column_labels={
                "item": "Assumption / limitation",
            },
            column_order=["item"],
        ),
    },
)

# ==========================================================
# Guardrails
# ==========================================================

def evaluate_chi_square_guardrails(run: Dict[str, Any]) -> list[Dict[str, Any]]:
    findings: list[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}

    cramers_v = metrics.get("cramers_v")
    cramers_v_magnitude = metrics.get("cramers_v_magnitude")
    expected_lt_5 = metrics.get("expected_cells_lt_5")
    chi_sig = metrics.get("significant_at_0_05")
    fisher_sig = metrics.get("fisher_exact_significant_at_0_05")
    fisher_p = metrics.get("fisher_exact_p_value")
    fisher_applicable = metrics.get("fisher_exact_applicable")

    if cramers_v is not None:
        findings.append(_new_finding(
            category="effect_size",
            severity="info",
            title=f"Cramer's V effect size: {cramers_v_magnitude or 'magnitude n/a'}",
            message=(
                f"The chi-square association has Cramer's V = {cramers_v}. "
                "Cramer's V is bounded in [0, 1] and quantifies the strength of association "
                "independent of sample size."
            ),
            evidence={
                "cramers_v": cramers_v,
                "cramers_v_magnitude": cramers_v_magnitude,
            },
        ))

    try:
        if expected_lt_5 is not None and int(expected_lt_5) > 0:
            findings.append(_new_finding(
                category="assumption_check",
                severity="warning",
                title=f"{expected_lt_5} expected cell(s) below 5",
                message=(
                    "The chi-square approximation assumes all expected cell counts are at least 5. "
                    f"{expected_lt_5} cell(s) violate this in the current table."
                ),
                evidence={"expected_cells_lt_5": expected_lt_5},
                recommendation=(
                    "For 2x2 tables, prefer Fisher's exact test (already reported). For larger "
                    "tables, combine sparse categories or collect more data."
                ),
            ))
    except Exception:
        pass

    if (
        fisher_applicable is True
        and chi_sig is not None
        and fisher_sig is not None
        and chi_sig != fisher_sig
    ):
        findings.append(_new_finding(
            category="interpretation",
            severity="warning",
            title="Chi-square and Fisher's exact disagree on significance",
            message=(
                "The chi-square approximation and Fisher's exact test give different verdicts "
                "at alpha=0.05. Because expected counts are low, Fisher's exact is the more "
                "trustworthy result."
            ),
            evidence={
                "chi_sig_at_0_05": chi_sig,
                "fisher_sig_at_0_05": fisher_sig,
                "fisher_p_value": fisher_p,
            },
            recommendation=(
                "Report and rely on Fisher's exact test for inference; cite the chi-square "
                "result only for context."
            ),
        ))

    return findings

PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_chi_square",
    display_name="Chi-square Test",
    requires_confirmation=False,
    is_inferential=True,
    argument_schema=ArgumentSchema(
        required={
            "row_col": str,
            "col_col": str,
        },
        optional={
            "yates_correction": bool,
        },
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
))