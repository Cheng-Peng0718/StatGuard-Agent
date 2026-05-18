from typing import Any, Dict, Tuple
import math
import warnings

import statsmodels.api as sm
import statsmodels.stats.api as sms

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
    safe_join_list,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.regression_utils import prepare_regression_data
from core.guardrails import _new_finding


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


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _p_value_interpretation(p_value: Any, alpha: float) -> str:
    p = _safe_float(p_value)

    if p is None:
        return "p-value not available"

    if p < alpha:
        return f"statistically significant at alpha={alpha}"

    return f"not statistically significant at alpha={alpha}"


def _coefficient_direction(coef: Any) -> str:
    value = _safe_float(coef)

    if value is None:
        return "unknown"

    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "zero"


def _is_intercept_term(term: Any) -> bool:
    lower = str(term).strip().lower()
    return lower in {"const", "intercept", "(intercept)"}


def _build_encoded_term_metadata(used_features: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata = {}

    for feature in used_features or []:
        if feature.get("type") != "categorical_encoded":
            continue

        column = feature.get("column")
        reference_level = feature.get("reference_level")
        encoded_level_map = feature.get("encoded_level_map") or {}

        for encoded_col in feature.get("encoded_columns", []) or []:
            encoded_col_str = str(encoded_col)
            level = encoded_level_map.get(encoded_col)

            if level is None:
                prefix = f"{column}_"
                if encoded_col_str.startswith(prefix):
                    level = encoded_col_str[len(prefix):]
                else:
                    level = encoded_col_str

            metadata[encoded_col_str] = {
                "feature": column,
                "level": level,
                "reference_level": reference_level,
                "variable_type": "categorical_dummy",
            }

    return metadata


def _format_abs_estimate(value: Any) -> str:
    numeric = _safe_float(value)

    if numeric is None:
        return "an unavailable amount"

    return f"{abs(numeric):.4g}"


def _build_coefficient_interpretations(
    coef_table: list[dict[str, Any]],
    target_col: str,
    alpha: float,
    used_features: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    encoded_term_metadata = _build_encoded_term_metadata(used_features or [])

    for row in coef_table:
        term = row.get("term")

        if _is_intercept_term(term):
            continue

        term_str = str(term)
        coef = row.get("coef")
        p_value = row.get("p_value")
        direction = _coefficient_direction(coef)
        significance = _p_value_interpretation(p_value, alpha)
        coef_value = _safe_float(coef)
        abs_estimate = _format_abs_estimate(coef)

        term_metadata = encoded_term_metadata.get(term_str)

        variable_type = "numeric_or_continuous"
        source_feature = term_str
        level = None
        reference_level = None

        if term_metadata:
            variable_type = "categorical_dummy"
            source_feature = term_metadata.get("feature") or term_str
            level = term_metadata.get("level")
            reference_level = term_metadata.get("reference_level")

            if coef_value is None:
                interpretation = (
                    f"The coefficient for `{source_feature} = {level}` could not be interpreted numerically."
                )
            else:
                comparison_text = (
                    f"Compared with the reference category"
                )
                if reference_level is not None:
                    comparison_text += f" `{source_feature} = {reference_level}`"
                comparison_text += f", `{source_feature} = {level}`"

                if direction == "positive":
                    interpretation = (
                        f"{comparison_text} is associated with an estimated {abs_estimate} higher "
                        f"`{target_col}`, holding other predictors fixed. This difference is {significance}."
                    )
                elif direction == "negative":
                    interpretation = (
                        f"{comparison_text} is associated with an estimated {abs_estimate} lower "
                        f"`{target_col}`, holding other predictors fixed. This difference is {significance}."
                    )
                else:
                    interpretation = (
                        f"{comparison_text} has an estimated coefficient near zero for `{target_col}`, "
                        f"holding other predictors fixed. This difference is {significance}."
                    )
        else:
            if coef_value is None:
                interpretation = (
                    f"The coefficient for `{term_str}` could not be interpreted numerically."
                )
            elif direction == "positive":
                interpretation = (
                    f"For a one-unit increase in `{term_str}`, `{target_col}` is estimated to increase by "
                    f"{abs_estimate}, holding other predictors fixed. This association is {significance}."
                )
            elif direction == "negative":
                interpretation = (
                    f"For a one-unit increase in `{term_str}`, `{target_col}` is estimated to decrease by "
                    f"{abs_estimate}, holding other predictors fixed. This association is {significance}."
                )
            else:
                interpretation = (
                    f"For a one-unit increase in `{term_str}`, `{target_col}` has an estimated change near zero, "
                    f"holding other predictors fixed. This association is {significance}."
                )

        rows.append({
            "term": term_str,
            "source_feature": source_feature,
            "variable_type": variable_type,
            "level": level,
            "reference_level": reference_level,
            "estimate": _round_or_none(coef),
            "direction": direction,
            "p_value": _round_or_none(p_value),
            "significance": significance,
            "ci_lower": row.get("ci_lower"),
            "ci_upper": row.get("ci_upper"),
            "interpretation": interpretation,
        })

    return rows


def _build_regression_assumptions_and_limitations() -> list[dict[str, str]]:
    items = [
        "The model describes association, not causation, unless the data come from a randomized or otherwise causal design.",
        "OLS assumes an approximately linear and additive relationship between predictors and the numeric outcome.",
        "OLS assumes independent observations.",
        "Standard errors, confidence intervals, and p-values rely on residual assumptions such as approximately constant variance and appropriate error behavior.",
        "Categorical predictors are interpreted relative to an omitted reference category after encoding.",
        "Multicollinearity, influential observations, autocorrelation, and residual normality should be checked with regression diagnostics when making formal conclusions.",
        "Heteroscedasticity-consistent (HC3) robust standard errors are computed automatically alongside classical OLS standard errors so that inference can be compared.",
    ]

    return [{"item": item} for item in items]


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


def _build_classical_coef_table(model, alpha: float) -> list[dict[str, Any]]:
    """Coefficient table using classical (non-robust) OLS standard errors."""
    conf = model.conf_int(alpha=alpha)

    rows = []

    for term in model.params.index:
        rows.append({
            "term": str(term),
            "coef": _round_or_none(model.params[term]),
            "std_err": _round_or_none(model.bse[term]),
            "t": _round_or_none(model.tvalues[term]),
            "p_value": _round_or_none(model.pvalues[term]),
            "ci_lower": _round_or_none(conf.loc[term, 0]) if term in conf.index else None,
            "ci_upper": _round_or_none(conf.loc[term, 1]) if term in conf.index else None,
        })

    return rows


def _build_robust_coef_table(robust_model, alpha: float) -> list[dict[str, Any]]:
    """Coefficient table using HC3 robust standard errors."""
    try:
        conf = robust_model.conf_int(alpha=alpha)
    except Exception:
        conf = None

    rows = []

    for term in robust_model.params.index:
        ci_lower = None
        ci_upper = None

        if conf is not None and term in conf.index:
            try:
                ci_lower = _round_or_none(conf.loc[term, 0])
                ci_upper = _round_or_none(conf.loc[term, 1])
            except Exception:
                ci_lower = None
                ci_upper = None

        rows.append({
            "term": str(term),
            "coef_robust": _round_or_none(robust_model.params[term]),
            "std_err_robust": _round_or_none(robust_model.bse[term]),
            "t_robust": _round_or_none(robust_model.tvalues[term]),
            "p_value_robust": _round_or_none(robust_model.pvalues[term]),
            "ci_lower_robust": ci_lower,
            "ci_upper_robust": ci_upper,
        })

    return rows


def _build_robust_comparison_table(
    classical: list[dict[str, Any]],
    robust: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    """
    Side-by-side comparison of classical and robust SE results.

    Useful when heteroscedasticity is suspected: differences in p-values or CIs
    between classical and HC3 robust columns indicate inference is sensitive to
    variance assumptions.
    """
    robust_by_term = {row["term"]: row for row in robust}

    rows = []

    for c in classical:
        term = c["term"]
        r = robust_by_term.get(term, {})

        rows.append({
            "term": term,
            "estimate": c.get("coef"),
            "std_err_classical": c.get("std_err"),
            "std_err_robust_hc3": r.get("std_err_robust"),
            "p_value_classical": c.get("p_value"),
            "p_value_robust_hc3": r.get("p_value_robust"),
            "ci_lower_classical": c.get("ci_lower"),
            "ci_upper_classical": c.get("ci_upper"),
            "ci_lower_robust_hc3": r.get("ci_lower_robust"),
            "ci_upper_robust_hc3": r.get("ci_upper_robust"),
            "inference_changed_at_alpha": _inference_changed(
                c.get("p_value"),
                r.get("p_value_robust"),
                alpha,
            ),
        })

    return rows


def _inference_changed(p_classical, p_robust, alpha: float):
    pc = _safe_float(p_classical)
    pr = _safe_float(p_robust)

    if pc is None or pr is None:
        return None

    sig_classical = pc < alpha
    sig_robust = pr < alpha

    return bool(sig_classical != sig_robust)


def execute_linear_model(context) -> Dict[str, Any]:
    """
    Fit an OLS multiple linear regression model.

    Args:
        target_col: numeric outcome column
        feature_cols: list of predictor columns
        max_missing_rate: optional, default 0.40
        max_categorical_levels: optional, default 10
        numeric_parse_threshold: optional, default 0.85
        min_n_per_parameter: optional, default 3
        alpha: optional, default 0.05
    """
    try:
        df = context.load_df()

        target_col = _get_arg(context, "target_col")
        feature_cols = _get_arg(context, "feature_cols", [])

        try:
            alpha = float(_get_arg(context, "alpha", 0.05))
        except Exception:
            alpha = 0.05

        if not (0 < alpha < 1):
            alpha = 0.05

        prep = prepare_regression_data(
            df,
            target_col,
            feature_cols,
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

        # Classical (non-robust) coefficient table
        coef_table = _build_classical_coef_table(model, alpha)

        # ----------------------------------------------------
        # HC3 robust SE
        # ----------------------------------------------------
        robust_results_available = False
        coef_table_robust: list[dict[str, Any]] = []
        coef_table_compare: list[dict[str, Any]] = []
        robust_summary: dict[str, Any] = {
            "available": False,
            "method": "HC3 (MacKinnon-White)",
        }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                robust_model = model.get_robustcov_results(cov_type="HC3")

            coef_table_robust = _build_robust_coef_table(robust_model, alpha)
            coef_table_compare = _build_robust_comparison_table(coef_table, coef_table_robust, alpha)

            n_changed = sum(
                1 for row in coef_table_compare
                if row.get("inference_changed_at_alpha") is True
                and not _is_intercept_term(row.get("term"))
            )

            robust_results_available = True
            robust_summary = {
                "available": True,
                "method": "HC3 (MacKinnon-White) heteroscedasticity-consistent standard errors",
                "n_predictors_with_changed_inference": int(n_changed),
                "note": (
                    "HC3 robust standard errors are reported alongside classical OLS standard errors. "
                    "When heteroscedasticity is present, robust inference is preferred."
                ),
            }
        except Exception:
            robust_results_available = False

        # ----------------------------------------------------
        # Coefficient interpretations
        # ----------------------------------------------------
        coefficient_interpretations = _build_coefficient_interpretations(
            coef_table=coef_table,
            target_col=target_col,
            alpha=alpha,
            used_features=prep["details"].get("used_features", []),
        )

        significant_predictor_count = sum(
            1
            for row in coef_table
            if not _is_intercept_term(row.get("term"))
            and _safe_float(row.get("p_value")) is not None
            and _safe_float(row.get("p_value")) < alpha
        )

        model_p_value = _safe_float(model.f_pvalue)
        model_significant_at_alpha = (
            model_p_value is not None and model_p_value < alpha
        )

        # ----------------------------------------------------
        # Inline Breusch-Pagan as quick heteroscedasticity flag
        # ----------------------------------------------------
        bp_flag = None

        try:
            _bp_stat, bp_pvalue, _bp_f, _bp_fp = sms.het_breuschpagan(
                model.resid,
                model.model.exog,
            )

            if math.isfinite(float(bp_pvalue)):
                bp_flag = bool(bp_pvalue < 0.05)
        except Exception:
            bp_flag = None

        # ----------------------------------------------------
        # Model spec for handoff to diagnostics
        # ----------------------------------------------------
        encoded_feature_cols = [str(col) for col in X.columns]

        model_spec = {
            "model_type": "ols",
            "target_col": str(target_col),
            "original_feature_cols": [str(col) for col in (feature_cols or [])],
            "encoded_feature_cols": encoded_feature_cols,
            "used_features": prep["details"].get("used_features", []),
            "excluded_features": prep["details"].get("excluded_features", []),
            "data_version_id": getattr(context, "active_data_version_id", None),
            "nobs": int(model.nobs),
            "r_squared": _round_or_none(model.rsquared),
            "adj_r_squared": _round_or_none(model.rsquared_adj),
        }

        details = {
            **prep["details"],
            "model_type": "OLS multiple linear regression",
            "model_spec": model_spec,
            "r_squared": _round_or_none(model.rsquared),
            "adj_r_squared": _round_or_none(model.rsquared_adj),
            "f_statistic": _round_or_none(model.fvalue),
            "f_p_value": _round_or_none(model.f_pvalue),
            "aic": _round_or_none(model.aic),
            "bic": _round_or_none(model.bic),
            "nobs": int(model.nobs),
            "df_model": _round_or_none(model.df_model),
            "df_resid": _round_or_none(model.df_resid),
            "coef_table": coef_table,
            "coef_table_robust_hc3": coef_table_robust if robust_results_available else [],
            "coef_table_classical_vs_robust": coef_table_compare if robust_results_available else [],
            "robust_se_summary": robust_summary,
            "alpha": alpha,
            "model_significant_at_alpha": bool(model_significant_at_alpha),
            "significant_predictor_count": int(significant_predictor_count),
            "coefficient_interpretations": coefficient_interpretations,
            "heteroscedasticity_flag_breusch_pagan_0_05": bp_flag,
            "assumptions_and_limitations": _build_regression_assumptions_and_limitations(),
        }

        status = "ok"
        message = "Multiple regression completed successfully."

        if (
            int(model.nobs) < 30
            or prep["details"]["n_eff"] < 5 * (prep["details"]["p_eff"] + 1)
        ):
            status = "warning"
            message = (
                "Multiple regression completed, but sample size is small relative "
                "to the number of predictors. Interpret cautiously."
            )

        if bp_flag is True and status != "warning":
            status = "warning"
            message = (
                "Multiple regression completed; Breusch-Pagan suggests heteroscedasticity. "
                "Prefer the HC3 robust standard errors reported alongside classical OLS."
            )

        if status == "warning":
            return _warning(message, details)

        return _ok(message, details)

    except Exception as e:
        return _failed(
            "OLS_FIT_EXCEPTION",
            "OLS model fitting failed.",
            e,
        )


def extract_linear_model(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target = arguments.get("target_col") or payload.get("target")
    features = arguments.get("feature_cols") or []
    feature_text = safe_join_list(features)

    if target and feature_text:
        title = f"Linear Model: {target} ~ {feature_text}"
    else:
        title = "Linear Model"

    robust_summary = payload.get("robust_se_summary", {}) or {}

    metrics = compact_dict({
        "nobs": payload.get("nobs"),
        "r_squared": payload.get("r_squared"),
        "adj_r_squared": payload.get("adj_r_squared"),
        "f_statistic": payload.get("f_statistic"),
        "f_p_value": payload.get("f_p_value"),
        "alpha": payload.get("alpha"),
        "model_significant_at_alpha": payload.get("model_significant_at_alpha"),
        "significant_predictor_count": payload.get("significant_predictor_count"),
        "heteroscedasticity_flag_breusch_pagan_0_05": payload.get("heteroscedasticity_flag_breusch_pagan_0_05"),
        "robust_se_available": robust_summary.get("available"),
        "n_predictors_with_changed_inference": robust_summary.get("n_predictors_with_changed_inference"),
    })

    tables: Dict[str, Any] = {}

    coef_table = payload.get("coef_table", [])
    if coef_table:
        tables["coef_table"] = coef_table

    coef_table_compare = payload.get("coef_table_classical_vs_robust", [])
    if coef_table_compare:
        tables["coef_table_classical_vs_robust"] = coef_table_compare

    coefficient_interpretations = payload.get("coefficient_interpretations", [])
    if coefficient_interpretations:
        tables["coefficient_interpretations"] = coefficient_interpretations

    assumptions_and_limitations = payload.get("assumptions_and_limitations", [])
    if assumptions_and_limitations:
        tables["assumptions_and_limitations"] = assumptions_and_limitations

    metadata = compact_dict({
        "model_type": payload.get("model_type"),
        "model_spec": payload.get("model_spec"),
        "aic": payload.get("aic"),
        "bic": payload.get("bic"),
        "df_model": payload.get("df_model"),
        "df_resid": payload.get("df_resid"),
        "n_eff": payload.get("n_eff"),
        "p_eff": payload.get("p_eff"),
        "target": payload.get("target"),
        "encoded_columns": payload.get("encoded_columns"),
        "used_features": payload.get("used_features"),
        "excluded_features": payload.get("excluded_features"),
        "raw_feature_count": payload.get("raw_feature_count"),
        "encoded_column_count": payload.get("encoded_column_count"),
        "min_required": payload.get("min_required"),
        "robust_se_summary": robust_summary,
        "coef_table_robust_hc3": payload.get("coef_table_robust_hc3"),
    })

    summary = "Fitted a linear model."

    if target:
        summary += f" Outcome: `{target}`."

    if feature_text:
        summary += f" Predictors: `{feature_text}`."

    if payload.get("nobs") is not None:
        summary += f" n={payload.get('nobs')}."

    if payload.get("r_squared") is not None:
        summary += f" R²={payload.get('r_squared')}."

    if payload.get("model_significant_at_alpha") is True:
        summary += " The overall model is statistically significant at the selected alpha level."
    elif payload.get("model_significant_at_alpha") is False:
        summary += " The overall model is not statistically significant at the selected alpha level."

    if payload.get("significant_predictor_count") is not None:
        summary += f" Significant non-intercept predictors: {payload.get('significant_predictor_count')}."

    if payload.get("heteroscedasticity_flag_breusch_pagan_0_05") is True:
        summary += " Breusch-Pagan suggests heteroscedasticity; HC3 robust standard errors are reported alongside classical OLS."

    return title, summary, metrics, tables, metadata


LINEAR_MODEL_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "nobs": "Observations used",
            "r_squared": "R-squared",
            "adj_r_squared": "Adjusted R-squared",
            "f_statistic": "F statistic",
            "f_p_value": "Model p-value",
            "alpha": "Alpha",
            "model_significant_at_alpha": "Overall model significant",
            "significant_predictor_count": "Significant non-intercept predictors",
            "heteroscedasticity_flag_breusch_pagan_0_05": "Heteroscedasticity flag (BP)",
            "robust_se_available": "HC3 robust SE available",
            "n_predictors_with_changed_inference": "Predictors with classical vs HC3 inference change",
        },
        formatters={
            "r_squared": lambda x: format_number(x, digits=4),
            "adj_r_squared": lambda x: format_number(x, digits=4),
            "f_statistic": lambda x: format_number(x, digits=4),
            "f_p_value": format_p_value,
            "alpha": lambda x: format_number(x, digits=4),
            "model_significant_at_alpha": format_bool_yes_no,
            "heteroscedasticity_flag_breusch_pagan_0_05": format_bool_yes_no,
            "robust_se_available": format_bool_yes_no,
        },
        order=[
            "nobs",
            "r_squared",
            "adj_r_squared",
            "f_statistic",
            "f_p_value",
            "alpha",
            "model_significant_at_alpha",
            "significant_predictor_count",
            "heteroscedasticity_flag_breusch_pagan_0_05",
            "robust_se_available",
            "n_predictors_with_changed_inference",
        ],
    ),
    tables={
        "coefficient_interpretations": TableDisplayConfig(
            column_labels={
                "term": "Term",
                "source_feature": "Source feature",
                "variable_type": "Variable type",
                "level": "Level",
                "reference_level": "Reference level",
                "estimate": "Estimate",
                "direction": "Direction",
                "p_value": "p-value",
                "significance": "Significance",
                "ci_lower": "95% CI lower",
                "ci_upper": "95% CI upper",
                "interpretation": "Interpretation",
            },
            column_order=[
                "term",
                "source_feature",
                "variable_type",
                "level",
                "reference_level",
                "estimate",
                "direction",
                "p_value",
                "significance",
                "ci_lower",
                "ci_upper",
                "interpretation",
            ],
            column_formatters={
                "estimate": lambda x: format_number(x, digits=4),
                "p_value": format_p_value,
                "ci_lower": lambda x: format_number(x, digits=4),
                "ci_upper": lambda x: format_number(x, digits=4),
            },
        ),
        "assumptions_and_limitations": TableDisplayConfig(
            column_labels={
                "item": "Assumption / limitation",
            },
            column_order=["item"],
        ),
        "coef_table": TableDisplayConfig(
            column_labels={
                "term": "Term",
                "coef": "Estimate",
                "std_err": "Std. Error",
                "t": "t",
                "p_value": "p-value",
                "ci_lower": "95% CI lower",
                "ci_upper": "95% CI upper",
            },
            column_formatters={
                "coef": lambda x: format_number(x, digits=4),
                "std_err": lambda x: format_number(x, digits=4),
                "t": lambda x: format_number(x, digits=4),
                "p_value": format_p_value,
                "ci_lower": lambda x: format_number(x, digits=4),
                "ci_upper": lambda x: format_number(x, digits=4),
            },
            column_order=[
                "term",
                "coef",
                "std_err",
                "t",
                "p_value",
                "ci_lower",
                "ci_upper",
            ],
            value_mappers={
                "term": {
                    "const": "Intercept",
                    "intercept": "Intercept",
                    "Intercept": "Intercept",
                }
            },
        ),
        "coef_table_classical_vs_robust": TableDisplayConfig(
            column_labels={
                "term": "Term",
                "estimate": "Estimate",
                "std_err_classical": "SE (classical)",
                "std_err_robust_hc3": "SE (HC3 robust)",
                "p_value_classical": "p (classical)",
                "p_value_robust_hc3": "p (HC3 robust)",
                "ci_lower_classical": "CI lower (classical)",
                "ci_upper_classical": "CI upper (classical)",
                "ci_lower_robust_hc3": "CI lower (HC3)",
                "ci_upper_robust_hc3": "CI upper (HC3)",
                "inference_changed_at_alpha": "Inference changed",
            },
            column_formatters={
                "estimate": lambda x: format_number(x, digits=4),
                "std_err_classical": lambda x: format_number(x, digits=4),
                "std_err_robust_hc3": lambda x: format_number(x, digits=4),
                "p_value_classical": format_p_value,
                "p_value_robust_hc3": format_p_value,
                "ci_lower_classical": lambda x: format_number(x, digits=4),
                "ci_upper_classical": lambda x: format_number(x, digits=4),
                "ci_lower_robust_hc3": lambda x: format_number(x, digits=4),
                "ci_upper_robust_hc3": lambda x: format_number(x, digits=4),
                "inference_changed_at_alpha": format_bool_yes_no,
            },
            column_order=[
                "term",
                "estimate",
                "std_err_classical",
                "std_err_robust_hc3",
                "p_value_classical",
                "p_value_robust_hc3",
                "ci_lower_classical",
                "ci_upper_classical",
                "ci_lower_robust_hc3",
                "ci_upper_robust_hc3",
                "inference_changed_at_alpha",
            ],
            value_mappers={
                "term": {
                    "const": "Intercept",
                    "intercept": "Intercept",
                    "Intercept": "Intercept",
                }
            },
        ),
    },
)

# ==========================================================
# Guardrails
# ==========================================================

def evaluate_regression_guardrails(run: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Guardrails for a fitted linear/regression-type model.

    Uses:
    - run["metrics"] for user-facing model metrics
    - run["metadata"] for internal fields such as p_eff / n_eff
    """
    findings: list[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}
    metadata = run.get("metadata", {}) or {}
    tables = run.get("tables", {}) or {}
    arguments = run.get("arguments", {}) or {}

    nobs = metrics.get("nobs", metadata.get("nobs"))
    r2 = metrics.get("r_squared")
    adj_r2 = metrics.get("adj_r_squared")
    f_p = metrics.get("f_p_value")

    p_eff = metadata.get("p_eff", metrics.get("p_eff"))
    n_eff = metadata.get("n_eff", metrics.get("n_eff"))

    target = arguments.get("target_col")
    features = arguments.get("feature_cols", [])

    # Sample size / model complexity
    if nobs is not None and p_eff is not None:
        try:
            nobs_f = float(nobs)
            p_eff_f = float(p_eff)

            if p_eff_f > 0:
                ratio = nobs_f / p_eff_f

                if ratio < 10:
                    findings.append(_new_finding(
                        category="sample_size",
                        severity="warning",
                        title="Low observations-per-parameter ratio",
                        message=(
                            "The model may be underpowered or unstable because the number "
                            "of observations per effective predictor is low."
                        ),
                        evidence={
                            "nobs": nobs,
                            "n_eff": n_eff,
                            "p_eff": p_eff,
                            "nobs_per_parameter": ratio,
                        },
                        recommendation=(
                            "Consider reducing model complexity, collecting more data, "
                            "or using regularized methods."
                        ),
                    ))
                else:
                    findings.append(_new_finding(
                        category="sample_size",
                        severity="info",
                        title="Sample size appears adequate for model size",
                        message=(
                            "The observations-per-parameter ratio does not raise an immediate "
                            "sample-size warning."
                        ),
                        evidence={
                            "nobs": nobs,
                            "n_eff": n_eff,
                            "p_eff": p_eff,
                            "nobs_per_parameter": ratio,
                        },
                    ))
        except Exception:
            pass

    # Explanatory power
    if r2 is not None:
        try:
            r2_f = float(r2)

            if r2_f < 0.10:
                findings.append(_new_finding(
                    category="model_fit",
                    severity="warning",
                    title="Low explanatory power",
                    message=(
                        "The model explains only a small fraction of outcome variation. "
                        "A statistically significant predictor may still have limited "
                        "practical predictive value."
                    ),
                    evidence={"r_squared": r2, "adj_r_squared": adj_r2},
                    recommendation=(
                        "Consider adding theoretically relevant predictors, checking nonlinear "
                        "relationships, or evaluating prediction error."
                    ),
                ))
            elif r2_f < 0.30:
                findings.append(_new_finding(
                    category="model_fit",
                    severity="info",
                    title="Moderate-to-low explanatory power",
                    message=(
                        "The model explains some variation, but substantial unexplained "
                        "variation remains."
                    ),
                    evidence={"r_squared": r2, "adj_r_squared": adj_r2},
                ))
            else:
                findings.append(_new_finding(
                    category="model_fit",
                    severity="info",
                    title="Model explains a nontrivial share of variation",
                    message=(
                        "The R-squared value suggests the model captures a meaningful share "
                        "of variation in the outcome."
                    ),
                    evidence={"r_squared": r2, "adj_r_squared": adj_r2},
                ))
        except Exception:
            pass

    # Significance vs causality
    if f_p is not None:
        try:
            f_p_f = float(f_p)

            if f_p_f < 0.05:
                findings.append(_new_finding(
                    category="interpretation",
                    severity="info",
                    title="Statistically significant association",
                    message=(
                        "The overall model is statistically significant. This supports an "
                        "association between the predictor set and the outcome, but does not "
                        "establish causality."
                    ),
                    evidence={
                        "f_p_value": f_p,
                        "target_col": target,
                        "feature_cols": features,
                    },
                    recommendation=(
                        "Avoid causal language unless the study design supports causal inference."
                    ),
                ))
        except Exception:
            pass

    # Inline heteroscedasticity flag from linear_model
    bp_flag_inline = metrics.get("heteroscedasticity_flag_breusch_pagan_0_05")

    if bp_flag_inline is True:
        n_changed = metrics.get("n_predictors_with_changed_inference")

        message = (
            "Breusch-Pagan suggests heteroscedasticity. HC3 robust standard errors are "
            "reported alongside classical OLS."
        )

        if n_changed is not None and int(n_changed) > 0:
            message += (
                f" For {n_changed} predictor(s), classical and HC3 robust inference "
                f"disagree at the chosen alpha; prefer the robust column."
            )

        findings.append(_new_finding(
            category="heteroscedasticity",
            severity="warning",
            title="Heteroscedasticity flagged in fitted model",
            message=message,
            evidence={
                "heteroscedasticity_flag_breusch_pagan_0_05": bp_flag_inline,
                "n_predictors_with_changed_inference": n_changed,
            },
            recommendation=(
                "Report HC3 robust standard errors. Consider transforming the outcome, "
                "respecifying the model, or weighted least squares if appropriate."
            ),
        ))

    # Coefficient table interpretation
    coef_table = tables.get("coef_table", []) or []

    if coef_table:
        for row in coef_table:
            if not isinstance(row, dict):
                continue

            term = row.get("term")

            if term in {"const", "intercept", "Intercept"}:
                continue

            p_value = row.get("p_value")
            coef = row.get("coef")

            try:
                p_f = float(p_value)
                coef_f = float(coef)

                if p_f < 0.05:
                    direction = "positive" if coef_f > 0 else "negative"

                    findings.append(_new_finding(
                        category="coefficient_interpretation",
                        severity="info",
                        title=f"Significant {direction} coefficient: {term}",
                        message=(
                            f"The coefficient for `{term}` is statistically significant "
                            f"and {direction}. Interpret this as an association conditional "
                            "on the model specification."
                        ),
                        evidence={
                            "term": term,
                            "coef": coef,
                            "p_value": p_value,
                            "ci_lower": row.get("ci_lower"),
                            "ci_upper": row.get("ci_upper"),
                        },
                    ))
            except Exception:
                pass

    return findings

PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_multiple_regression",
    display_name="Linear Model",
    evidence_categories=["regression_model", "statistical_inference"],
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
            "alpha": float,
        },
        column_args=[
            "target_col",
        ],
        column_list_args=[
            "feature_cols",
        ],
        allow_all_columns=False,
    ),
    execute=execute_linear_model,
    extractor=extract_linear_model,
    guardrail_evaluators=[
        evaluate_regression_guardrails,
    ],
    display_config=LINEAR_MODEL_DISPLAY,
))