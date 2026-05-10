from typing import Any, Dict, Tuple
import math
import warnings

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
from core.analysis_tool_plugins.shared.regression_utils import prepare_regression_data
from core.guardrails import evaluate_diagnostics_guardrails

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


def _warning(message: str, details: Dict[str, Any], artifacts=None):
    return {
        "status": "warning",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }


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


def execute_regression_diagnostics(context) -> Dict[str, Any]:
    """
    Run VIF and Breusch-Pagan diagnostics using the same prepared design matrix as OLS.

    Args:
        target_col: numeric outcome column
        feature_cols: list of predictor columns
        max_missing_rate: optional, default 0.40
        max_categorical_levels: optional, default 10
        numeric_parse_threshold: optional, default 0.85
        min_n_per_parameter: optional, default 3
    """
    try:
        df = context.load_df()

        prep = prepare_regression_data(
            df,
            _get_arg(context, "target_col"),
            _get_arg(context, "feature_cols", []),
            max_missing_rate=float(_get_arg(context, "max_missing_rate", 0.40)),
            max_categorical_levels=int(_get_arg(context, "max_categorical_levels", 10)),
            numeric_parse_threshold=float(_get_arg(context, "numeric_parse_threshold", 0.85)),
            min_n_per_parameter=int(_get_arg(context, "min_n_per_parameter", 3)),
        )

        if prep.get("status") != "ok":
            return prep

        y = prep["y"]
        X = prep["X"]
        X_const = sm.add_constant(X, has_constant="add")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X_const).fit()

        vif_rows = []

        for i, col in enumerate(X_const.columns):
            if col == "const":
                continue

            try:
                value = variance_inflation_factor(X_const.values, i)
                vif_value = _round_or_none(value)
            except Exception:
                vif_value = None

            vif_rows.append({
                "term": str(col),
                "vif": vif_value,
                "flag": bool(vif_value is not None and vif_value > 10),
            })

        bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = sms.het_breuschpagan(
            model.resid,
            model.model.exog,
        )

        breusch_pagan = {
            "lm_statistic": _round_or_none(bp_stat),
            "lm_p_value": _round_or_none(bp_pvalue),
            "f_statistic": _round_or_none(bp_fstat),
            "f_p_value": _round_or_none(bp_fpvalue),
            "heteroscedasticity_flag_0_05": (
                bool(bp_pvalue < 0.05)
                if math.isfinite(float(bp_pvalue))
                else None
            ),
        }

        details = {
            **prep["details"],
            "vif": vif_rows,
            "breusch_pagan": breusch_pagan,
        }

        has_vif_warning = any(row.get("flag") for row in vif_rows)
        has_bp_warning = breusch_pagan.get("heteroscedasticity_flag_0_05") is True

        if has_vif_warning or has_bp_warning:
            return _warning(
                "Regression diagnostics completed with statistical warnings.",
                details,
            )

        return _ok(
            "Regression diagnostics completed successfully.",
            details,
        )

    except Exception as e:
        return _failed(
            "REGRESSION_DIAGNOSTICS_EXCEPTION",
            "Regression diagnostics failed.",
            e,
        )


def extract_regression_diagnostics(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Model Diagnostics"

    vif = payload.get("vif", []) or []
    bp = payload.get("breusch_pagan", {}) or {}

    vif_values = [
        row.get("vif")
        for row in vif
        if isinstance(row, dict) and row.get("vif") is not None
    ]

    metrics = compact_dict({
        "max_vif": max(vif_values) if vif_values else None,
        "breusch_pagan_lm_statistic": bp.get("lm_statistic") if isinstance(bp, dict) else None,
        "breusch_pagan_lm_p_value": bp.get("lm_p_value") if isinstance(bp, dict) else None,
        "breusch_pagan_f_statistic": bp.get("f_statistic") if isinstance(bp, dict) else None,
        "breusch_pagan_f_p_value": bp.get("f_p_value") if isinstance(bp, dict) else None,
        "heteroscedasticity_flag_0_05": (
            bp.get("heteroscedasticity_flag_0_05")
            if isinstance(bp, dict)
            else None
        ),
    })

    tables: Dict[str, Any] = {}

    if vif:
        tables["vif"] = vif

    # Do not put raw breusch_pagan JSON into report tables.
    # The user-facing BP results are already in metrics.

    metadata = compact_dict({
        "breusch_pagan": bp,
        "n_eff": payload.get("n_eff"),
        "p_eff": payload.get("p_eff"),
        "target": payload.get("target"),
        "encoded_columns": payload.get("encoded_columns"),
        "used_features": payload.get("used_features"),
        "excluded_features": payload.get("excluded_features"),
        "raw_feature_count": payload.get("raw_feature_count"),
        "encoded_column_count": payload.get("encoded_column_count"),
        "min_required": payload.get("min_required"),
    })

    summary = "Computed multicollinearity and heteroscedasticity diagnostics."

    if metrics.get("max_vif") is not None:
        summary += f" Max VIF={metrics.get('max_vif')}."

    if metrics.get("breusch_pagan_lm_p_value") is not None:
        summary += f" Breusch-Pagan p={metrics.get('breusch_pagan_lm_p_value')}."

    return title, summary, metrics, tables, metadata


MODEL_DIAGNOSTICS_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "max_vif": "Maximum VIF",
            "breusch_pagan_lm_statistic": "Breusch-Pagan LM statistic",
            "breusch_pagan_lm_p_value": "Breusch-Pagan LM p-value",
            "breusch_pagan_f_statistic": "Breusch-Pagan F statistic",
            "breusch_pagan_f_p_value": "Breusch-Pagan F-test p-value",
            "heteroscedasticity_flag_0_05": "Heteroscedasticity flag",
        },
        formatters={
            "max_vif": lambda x: format_number(x, digits=4),
            "breusch_pagan_lm_statistic": lambda x: format_number(x, digits=4),
            "breusch_pagan_lm_p_value": format_p_value,
            "breusch_pagan_f_statistic": lambda x: format_number(x, digits=4),
            "breusch_pagan_f_p_value": format_p_value,
            "heteroscedasticity_flag_0_05": format_bool_yes_no,
        },
        order=[
            "max_vif",
            "breusch_pagan_lm_statistic",
            "breusch_pagan_lm_p_value",
            "breusch_pagan_f_statistic",
            "breusch_pagan_f_p_value",
            "heteroscedasticity_flag_0_05",
        ],
    ),
    tables={
        "vif": TableDisplayConfig(
            column_labels={
                "term": "Term",
                "vif": "VIF",
                "flag": "High VIF flag",
            },
            column_formatters={
                "vif": lambda x: format_number(x, digits=4),
                "flag": format_bool_yes_no,
            },
            column_order=[
                "term",
                "vif",
                "flag",
            ],
        ),
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="regression_diagnostics",
    display_name="Model Diagnostics",
    requires_confirmation=False,

    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "feature_cols": list,
        },
        optional={
            "max_missing_rate": float,
            "max_categorical_levels": int,
            "numeric_parse_threshold": float,
            "min_n_per_parameter": int,
        },
        column_args=[
            "target_col",
        ],
        column_list_args=[
            "feature_cols",
        ],
        allow_all_columns=False,
    ),

    execute=execute_regression_diagnostics,
    extractor=extract_regression_diagnostics,
    guardrail_evaluators=[
        evaluate_diagnostics_guardrails,
    ],
    display_config=MODEL_DIAGNOSTICS_DISPLAY,

    # Generic method/planning contract.
    method_family="model_diagnostics",

    # Regression diagnostics requires the same variable roles as the fitted
    # linear model: a continuous target and one or more predictors.
    # It should not auto-select variables from a generic plan.
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
                "Continuous numeric response variable for the diagnostic model."
            ),
        ),
        VariableRoleSpec(
            role_name="feature_cols",
            required=True,
            user_must_select=True,
            allowed_semantic_types=[
                "continuous_numeric",
                "discrete_numeric",
                "binary_categorical",
                "nominal_categorical",
                "ordinal_categorical",
            ],
            min_variables=1,
            max_variables=None,
            allow_auto_select=False,
            description=(
                "Predictor variables used in the diagnostic model. "
                "They should match the intended or previously fitted regression specification."
            ),
        ),
    ],

    planning_policy=NEEDS_USER_VARIABLES_PLANNING,

    planning_metadata=PlanningMetadata(
        supported_goal_types=[
            "regression_modeling",
        ],
        not_recommended_for_goal_types=[
            "dataset_overview",
            "analysis_recommendation",
            "analysis_planning",
        ],
        planning_tags=[
            "regression",
            "diagnostics",
            "model_checking",
        ],
        default_plan_purpose="Check model diagnostics after the regression fit.",
        expected_deliverables=[
            "regression_diagnostics",
        ],
        task_argument_bindings=[
            {
                "task_field": "target_variables",
                "index": 0,
                "argument": "target_col",
                "required_choice": "target_col",
            },
            {
                "task_field": "predictor_variables",
                "argument": "feature_cols",
                "required_choice": "feature_cols",
            },
        ],
        plan_order=20,
    ),

    # Regression diagnostics does not mutate data.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    repair_policy=DEFAULT_ANALYSIS_REPAIR,
))