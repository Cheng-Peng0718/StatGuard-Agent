from typing import Any, Dict, Tuple
import math
import warnings

import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.stattools import durbin_watson
from scipy import stats as scipy_stats

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


def _get_analysis_runs(context) -> list[dict[str, Any]]:
    runs = getattr(context, "analysis_runs", None)

    if runs is None:
        return []

    if isinstance(runs, list):
        return runs

    return []


def _run_id(run: dict[str, Any]) -> str | None:
    return run.get("run_id") or run.get("analysis_run_id")


def _extract_model_spec_from_run(run: dict[str, Any]) -> dict[str, Any] | None:
    metadata = run.get("metadata", {}) or {}

    model_spec = metadata.get("model_spec")

    if isinstance(model_spec, dict):
        return model_spec

    return None


def _find_regression_run_by_id(
    analysis_runs: list[dict[str, Any]],
    source_analysis_run_id: str | None,
) -> dict[str, Any] | None:
    if not source_analysis_run_id:
        return None

    for run in analysis_runs or []:
        if _run_id(run) == source_analysis_run_id:
            return run

    return None


def _find_latest_regression_run(
    analysis_runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for run in reversed(analysis_runs or []):
        if run.get("status") not in {"ok", "warning"}:
            continue

        categories = run.get("evidence_categories", []) or []

        if "regression_model" not in categories:
            continue

        if _extract_model_spec_from_run(run):
            return run

    return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, tuple):
        return [str(item) for item in value]

    if isinstance(value, str):
        return [value]

    return [str(value)]


def _has_missing_dataset_columns(
    df,
    target_col: Any,
    feature_cols: list[str],
) -> bool:
    if not target_col or target_col not in df.columns:
        return True

    return any(col not in df.columns for col in feature_cols)


def _resolve_diagnostic_model_inputs(
    context,
    df,
) -> tuple[str | None, list[str], dict[str, Any]]:
    """
    Resolve diagnostics target/features.

    Priority:
    1. Explicit source_analysis_run_id if provided.
    2. Latest successful regression_model run if explicit args are missing
       or explicit feature columns are not active dataset columns.
    3. Explicit target_col / feature_cols fallback.
    """
    explicit_target = _get_arg(context, "target_col")
    explicit_features = _as_string_list(_get_arg(context, "feature_cols", []))
    source_analysis_run_id = _get_arg(context, "source_analysis_run_id")

    analysis_runs = _get_analysis_runs(context)

    selected_run = _find_regression_run_by_id(
        analysis_runs,
        source_analysis_run_id,
    )

    if selected_run is None:
        selected_run = _find_latest_regression_run(analysis_runs)

    explicit_args_missing_or_invalid = (
        not explicit_target
        or not explicit_features
        or _has_missing_dataset_columns(df, explicit_target, explicit_features)
    )

    if selected_run is not None and (
        source_analysis_run_id or explicit_args_missing_or_invalid
    ):
        model_spec = _extract_model_spec_from_run(selected_run) or {}

        target_col = model_spec.get("target_col")
        feature_cols = _as_string_list(
            model_spec.get("original_feature_cols")
        )

        if target_col and feature_cols:
            return target_col, feature_cols, {
                "resolved_from_model_spec": True,
                "source_analysis_run_id": _run_id(selected_run),
                "source_analysis_title": selected_run.get("title"),
                "source_model_spec": model_spec,
                "explicit_target_col": explicit_target,
                "explicit_feature_cols": explicit_features,
            }

    return explicit_target, explicit_features, {
        "resolved_from_model_spec": False,
        "source_analysis_run_id": source_analysis_run_id,
        "explicit_target_col": explicit_target,
        "explicit_feature_cols": explicit_features,
    }


# ==========================================================
# Influence diagnostics
# ==========================================================

def _compute_influence_diagnostics(
    model,
    X_const,
    cooks_d_top_k: int = 10,
) -> dict[str, Any]:
    """
    Cook's distance, leverage (hat values), DFFITS, and studentized residuals.

    Returns summary statistics, threshold flags, and the top-k most influential
    observations by Cook's distance.
    """
    try:
        influence = OLSInfluence(model)
    except Exception:
        return {
            "available": False,
            "reason": "OLSInfluence failed to initialize.",
        }

    try:
        cooks_d = np.asarray(influence.cooks_distance[0], dtype=float)
        leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
        dffits = np.asarray(influence.dffits[0], dtype=float)

        # Internally studentized residuals (standardized residuals)
        # Externally studentized = studentized residuals (jackknife)
        student_resid = np.asarray(influence.resid_studentized_external, dtype=float)
    except Exception:
        return {
            "available": False,
            "reason": "Failed to compute influence measures.",
        }

    n = int(model.nobs)
    # Number of estimated parameters (including intercept)
    try:
        p_total = int(X_const.shape[1])
    except Exception:
        p_total = int(model.df_model) + 1

    # Standard thresholds
    cook_threshold = 4.0 / max(n, 1)
    leverage_threshold = 2.0 * p_total / max(n, 1)
    dffits_threshold = 2.0 * math.sqrt(max(p_total, 1) / max(n, 1))
    student_resid_threshold = 3.0

    # Flag counts
    n_high_cook = int(np.sum(np.isfinite(cooks_d) & (cooks_d > cook_threshold)))
    n_high_leverage = int(np.sum(np.isfinite(leverage) & (leverage > leverage_threshold)))
    n_high_dffits = int(np.sum(np.isfinite(dffits) & (np.abs(dffits) > dffits_threshold)))
    n_extreme_resid = int(
        np.sum(np.isfinite(student_resid) & (np.abs(student_resid) > student_resid_threshold))
    )

    # Top-k most influential rows by Cook's distance
    top_rows = []

    try:
        # Use original observation index from the model
        obs_index = list(model.model.data.row_labels) if hasattr(model.model, "data") else list(range(n))
    except Exception:
        obs_index = list(range(n))

    finite_mask = np.isfinite(cooks_d)
    finite_idx = np.where(finite_mask)[0]

    if finite_idx.size > 0:
        sorted_idx = finite_idx[np.argsort(-cooks_d[finite_idx])]
        top_k = sorted_idx[:cooks_d_top_k]

        for i in top_k:
            row_label = obs_index[i] if i < len(obs_index) else int(i)

            try:
                row_label = (
                    int(row_label) if isinstance(row_label, (int, np.integer))
                    else str(row_label)
                )
            except Exception:
                row_label = str(row_label)

            top_rows.append({
                "observation_index": row_label,
                "cooks_distance": _round_or_none(cooks_d[i]),
                "leverage": _round_or_none(leverage[i]),
                "dffits": _round_or_none(dffits[i]),
                "studentized_residual": _round_or_none(student_resid[i]),
                "flagged_high_cook": bool(cooks_d[i] > cook_threshold),
                "flagged_high_leverage": (
                    bool(leverage[i] > leverage_threshold)
                    if np.isfinite(leverage[i]) else False
                ),
                "flagged_high_dffits": (
                    bool(abs(dffits[i]) > dffits_threshold)
                    if np.isfinite(dffits[i]) else False
                ),
                "flagged_extreme_residual": (
                    bool(abs(student_resid[i]) > student_resid_threshold)
                    if np.isfinite(student_resid[i]) else False
                ),
            })

    # Summary statistics
    def _safe_max_abs(arr):
        if arr.size == 0:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(np.max(np.abs(finite)))

    def _safe_max(arr):
        if arr.size == 0:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(np.max(finite))

    return {
        "available": True,
        "n_observations": n,
        "n_parameters_with_intercept": p_total,
        "thresholds": {
            "cooks_distance": _round_or_none(cook_threshold),
            "cooks_distance_rule": "4 / n",
            "leverage": _round_or_none(leverage_threshold),
            "leverage_rule": "2 * p / n",
            "dffits_abs": _round_or_none(dffits_threshold),
            "dffits_rule": "2 * sqrt(p / n)",
            "studentized_residual_abs": student_resid_threshold,
            "studentized_residual_rule": "|t| > 3 (approx.)",
        },
        "summary": {
            "max_cooks_distance": _round_or_none(_safe_max(cooks_d)),
            "max_leverage": _round_or_none(_safe_max(leverage)),
            "max_abs_dffits": _round_or_none(_safe_max_abs(dffits)),
            "max_abs_studentized_residual": _round_or_none(_safe_max_abs(student_resid)),
        },
        "flag_counts": {
            "n_high_cooks_distance": n_high_cook,
            "n_high_leverage": n_high_leverage,
            "n_high_dffits": n_high_dffits,
            "n_extreme_studentized_residual": n_extreme_resid,
        },
        "top_influential_observations": top_rows,
    }


def _compute_residual_normality(resid: np.ndarray) -> dict[str, Any]:
    """
    Residual normality assessment using Shapiro-Wilk and/or Jarque-Bera.

    Shapiro-Wilk is preferred for n <= 5000. Jarque-Bera is large-sample-friendly
    and is always reported when n >= 8.
    """
    out: dict[str, Any] = {
        "shapiro_wilk": None,
        "jarque_bera": None,
    }

    n = int(resid.size)

    # Shapiro-Wilk: best for n in [3, 5000]
    if 3 <= n <= 5000:
        try:
            stat_sw, p_sw = scipy_stats.shapiro(resid)
            out["shapiro_wilk"] = {
                "statistic": _round_or_none(stat_sw),
                "p_value": _round_or_none(p_sw),
                "normal_at_0_05": (
                    bool(p_sw >= 0.05) if math.isfinite(float(p_sw)) else None
                ),
                "note": None,
            }
        except Exception:
            out["shapiro_wilk"] = {
                "statistic": None,
                "p_value": None,
                "normal_at_0_05": None,
                "note": "Shapiro-Wilk failed.",
            }
    else:
        out["shapiro_wilk"] = {
            "statistic": None,
            "p_value": None,
            "normal_at_0_05": None,
            "note": (
                f"n={n} outside Shapiro-Wilk valid range [3, 5000]; "
                "Jarque-Bera reported instead."
            ),
        }

    # Jarque-Bera
    if n >= 8:
        try:
            stat_jb, p_jb = scipy_stats.jarque_bera(resid)
            out["jarque_bera"] = {
                "statistic": _round_or_none(stat_jb),
                "p_value": _round_or_none(p_jb),
                "normal_at_0_05": (
                    bool(p_jb >= 0.05) if math.isfinite(float(p_jb)) else None
                ),
                "note": None,
            }
        except Exception:
            out["jarque_bera"] = {
                "statistic": None,
                "p_value": None,
                "normal_at_0_05": None,
                "note": "Jarque-Bera failed.",
            }
    else:
        out["jarque_bera"] = {
            "statistic": None,
            "p_value": None,
            "normal_at_0_05": None,
            "note": "n < 8; Jarque-Bera not recommended.",
        }

    # Composite verdict
    verdict_normal: Any = None
    sw_normal = (out["shapiro_wilk"] or {}).get("normal_at_0_05")
    jb_normal = (out["jarque_bera"] or {}).get("normal_at_0_05")

    if sw_normal is False or jb_normal is False:
        verdict_normal = False
    elif sw_normal is True or jb_normal is True:
        verdict_normal = True

    out["normality_verdict_at_0_05"] = verdict_normal

    return out


def _interpret_durbin_watson(dw: float) -> str:
    if not math.isfinite(dw):
        return "undefined"

    # Standard rules of thumb. Exact cutoffs depend on n and p; this is a heuristic.
    if dw < 1.5:
        return "positive autocorrelation possible"
    if dw > 2.5:
        return "negative autocorrelation possible"

    return "no strong autocorrelation"


# ==========================================================
# Main execute
# ==========================================================

def execute_regression_diagnostics(context) -> Dict[str, Any]:
    """
    Comprehensive regression diagnostics:
    - VIF (multicollinearity)
    - Breusch-Pagan (heteroscedasticity)
    - Durbin-Watson (autocorrelation)
    - Shapiro-Wilk / Jarque-Bera (residual normality)
    - Cook's distance, leverage, DFFITS, studentized residuals (influence)

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

        target_col, feature_cols, resolution_details = _resolve_diagnostic_model_inputs(
            context,
            df,
        )

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

        # ----------------------------------------------------
        # VIF
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Breusch-Pagan
        # ----------------------------------------------------
        try:
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
                    if math.isfinite(float(bp_pvalue)) else None
                ),
            }
        except Exception:
            breusch_pagan = {
                "lm_statistic": None,
                "lm_p_value": None,
                "f_statistic": None,
                "f_p_value": None,
                "heteroscedasticity_flag_0_05": None,
            }

        # ----------------------------------------------------
        # Durbin-Watson
        # ----------------------------------------------------
        try:
            dw_stat = float(durbin_watson(model.resid))
        except Exception:
            dw_stat = float("nan")

        durbin_watson_result = {
            "statistic": _round_or_none(dw_stat),
            "interpretation": _interpret_durbin_watson(dw_stat),
            "note": (
                "Durbin-Watson statistic. Values close to 2 indicate no autocorrelation. "
                "Exact critical values depend on n and p; here we apply a heuristic 1.5/2.5 rule."
            ),
        }

        # ----------------------------------------------------
        # Residual normality
        # ----------------------------------------------------
        residual_normality = _compute_residual_normality(np.asarray(model.resid, dtype=float))

        # ----------------------------------------------------
        # Influence diagnostics
        # ----------------------------------------------------
        influence = _compute_influence_diagnostics(model, X_const, cooks_d_top_k=10)

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------
        details = {
            **prep["details"],
            **resolution_details,
            "vif": vif_rows,
            "breusch_pagan": breusch_pagan,
            "durbin_watson": durbin_watson_result,
            "residual_normality": residual_normality,
            "influence_diagnostics": influence,
        }

        # Determine status. Surface a "warning" if any of the following:
        has_vif_warning = any(row.get("flag") for row in vif_rows)
        has_bp_warning = breusch_pagan.get("heteroscedasticity_flag_0_05") is True

        flag_counts = (influence.get("flag_counts") or {}) if influence.get("available") else {}
        has_influence_warning = any([
            int(flag_counts.get("n_high_cooks_distance", 0)) > 0,
            int(flag_counts.get("n_high_leverage", 0)) > 0,
            int(flag_counts.get("n_high_dffits", 0)) > 0,
            int(flag_counts.get("n_extreme_studentized_residual", 0)) > 0,
        ])

        verdict_normal = (residual_normality or {}).get("normality_verdict_at_0_05")
        has_normality_warning = verdict_normal is False

        dw_warn = durbin_watson_result.get("interpretation") in {
            "positive autocorrelation possible",
            "negative autocorrelation possible",
        }

        any_warning = (
            has_vif_warning
            or has_bp_warning
            or has_influence_warning
            or has_normality_warning
            or dw_warn
        )

        if any_warning:
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


# ==========================================================
# Extractor / Display
# ==========================================================

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
    dw = payload.get("durbin_watson", {}) or {}
    rn = payload.get("residual_normality", {}) or {}
    influence = payload.get("influence_diagnostics", {}) or {}

    vif_values = [
        row.get("vif")
        for row in vif
        if isinstance(row, dict) and row.get("vif") is not None
    ]

    sw = rn.get("shapiro_wilk") or {}
    jb = rn.get("jarque_bera") or {}

    summary_block = influence.get("summary", {}) if influence.get("available") else {}
    flags_block = influence.get("flag_counts", {}) if influence.get("available") else {}

    metrics = compact_dict({
        "max_vif": max(vif_values) if vif_values else None,
        "breusch_pagan_lm_statistic": bp.get("lm_statistic"),
        "breusch_pagan_lm_p_value": bp.get("lm_p_value"),
        "breusch_pagan_f_statistic": bp.get("f_statistic"),
        "breusch_pagan_f_p_value": bp.get("f_p_value"),
        "heteroscedasticity_flag_0_05": bp.get("heteroscedasticity_flag_0_05"),
        "durbin_watson_statistic": dw.get("statistic"),
        "durbin_watson_interpretation": dw.get("interpretation"),
        "shapiro_wilk_statistic": sw.get("statistic"),
        "shapiro_wilk_p_value": sw.get("p_value"),
        "shapiro_wilk_normal_at_0_05": sw.get("normal_at_0_05"),
        "jarque_bera_statistic": jb.get("statistic"),
        "jarque_bera_p_value": jb.get("p_value"),
        "jarque_bera_normal_at_0_05": jb.get("normal_at_0_05"),
        "residuals_appear_normal_at_0_05": rn.get("normality_verdict_at_0_05"),
        "max_cooks_distance": summary_block.get("max_cooks_distance"),
        "max_leverage": summary_block.get("max_leverage"),
        "max_abs_dffits": summary_block.get("max_abs_dffits"),
        "max_abs_studentized_residual": summary_block.get("max_abs_studentized_residual"),
        "n_high_cooks_distance": flags_block.get("n_high_cooks_distance"),
        "n_high_leverage": flags_block.get("n_high_leverage"),
        "n_high_dffits": flags_block.get("n_high_dffits"),
        "n_extreme_studentized_residual": flags_block.get("n_extreme_studentized_residual"),
    })

    tables: Dict[str, Any] = {}

    if vif:
        tables["vif"] = vif

    top_rows = influence.get("top_influential_observations") if influence.get("available") else None
    if top_rows:
        tables["top_influential_observations"] = top_rows

    metadata = compact_dict({
        "breusch_pagan": bp,
        "durbin_watson": dw,
        "residual_normality": rn,
        "influence_diagnostics": influence,
        "resolved_from_model_spec": payload.get("resolved_from_model_spec"),
        "source_analysis_run_id": payload.get("source_analysis_run_id"),
        "source_analysis_title": payload.get("source_analysis_title"),
        "source_model_spec": payload.get("source_model_spec"),
        "explicit_target_col": payload.get("explicit_target_col"),
        "explicit_feature_cols": payload.get("explicit_feature_cols"),
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

    summary = "Computed full regression diagnostics suite (multicollinearity, heteroscedasticity, autocorrelation, residual normality, and influence)."

    if metrics.get("max_vif") is not None:
        summary += f" Max VIF={metrics.get('max_vif')}."

    if metrics.get("breusch_pagan_lm_p_value") is not None:
        summary += f" Breusch-Pagan p={metrics.get('breusch_pagan_lm_p_value')}."

    if metrics.get("durbin_watson_statistic") is not None:
        summary += f" Durbin-Watson={metrics.get('durbin_watson_statistic')}."

    if metrics.get("n_high_cooks_distance") is not None:
        summary += f" High-influence observations (Cook's D > 4/n): {metrics.get('n_high_cooks_distance')}."

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
            "durbin_watson_statistic": "Durbin-Watson statistic",
            "durbin_watson_interpretation": "Durbin-Watson interpretation",
            "shapiro_wilk_statistic": "Shapiro-Wilk W",
            "shapiro_wilk_p_value": "Shapiro-Wilk p-value",
            "shapiro_wilk_normal_at_0_05": "Residuals normal (Shapiro)",
            "jarque_bera_statistic": "Jarque-Bera statistic",
            "jarque_bera_p_value": "Jarque-Bera p-value",
            "jarque_bera_normal_at_0_05": "Residuals normal (JB)",
            "residuals_appear_normal_at_0_05": "Residuals appear normal (overall)",
            "max_cooks_distance": "Max Cook's distance",
            "max_leverage": "Max leverage",
            "max_abs_dffits": "Max |DFFITS|",
            "max_abs_studentized_residual": "Max |studentized residual|",
            "n_high_cooks_distance": "High Cook's-D observations",
            "n_high_leverage": "High-leverage observations",
            "n_high_dffits": "High-|DFFITS| observations",
            "n_extreme_studentized_residual": "Extreme studentized residuals",
        },
        formatters={
            "max_vif": format_number,
            "breusch_pagan_lm_statistic": format_number,
            "breusch_pagan_lm_p_value": format_p_value,
            "breusch_pagan_f_statistic": format_number,
            "breusch_pagan_f_p_value": format_p_value,
            "heteroscedasticity_flag_0_05": format_bool_yes_no,
            "durbin_watson_statistic": format_number,
            "shapiro_wilk_statistic": format_number,
            "shapiro_wilk_p_value": format_p_value,
            "shapiro_wilk_normal_at_0_05": format_bool_yes_no,
            "jarque_bera_statistic": format_number,
            "jarque_bera_p_value": format_p_value,
            "jarque_bera_normal_at_0_05": format_bool_yes_no,
            "residuals_appear_normal_at_0_05": format_bool_yes_no,
            "max_cooks_distance": format_number,
            "max_leverage": format_number,
            "max_abs_dffits": format_number,
            "max_abs_studentized_residual": format_number,
        },
        order=[
            "max_vif",
            "breusch_pagan_lm_statistic",
            "breusch_pagan_lm_p_value",
            "breusch_pagan_f_statistic",
            "breusch_pagan_f_p_value",
            "heteroscedasticity_flag_0_05",
            "durbin_watson_statistic",
            "durbin_watson_interpretation",
            "shapiro_wilk_statistic",
            "shapiro_wilk_p_value",
            "shapiro_wilk_normal_at_0_05",
            "jarque_bera_statistic",
            "jarque_bera_p_value",
            "jarque_bera_normal_at_0_05",
            "residuals_appear_normal_at_0_05",
            "max_cooks_distance",
            "max_leverage",
            "max_abs_dffits",
            "max_abs_studentized_residual",
            "n_high_cooks_distance",
            "n_high_leverage",
            "n_high_dffits",
            "n_extreme_studentized_residual",
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
                "vif": format_number,
                "flag": format_bool_yes_no,
            },
            column_order=[
                "term",
                "vif",
                "flag",
            ],
        ),
        "top_influential_observations": TableDisplayConfig(
            column_labels={
                "observation_index": "Observation",
                "cooks_distance": "Cook's D",
                "leverage": "Leverage",
                "dffits": "DFFITS",
                "studentized_residual": "Studentized residual",
                "flagged_high_cook": "High Cook's D",
                "flagged_high_leverage": "High leverage",
                "flagged_high_dffits": "High |DFFITS|",
                "flagged_extreme_residual": "Extreme residual",
            },
            column_formatters={
                "cooks_distance": format_number,
                "leverage": format_number,
                "dffits": format_number,
                "studentized_residual": format_number,
                "flagged_high_cook": format_bool_yes_no,
                "flagged_high_leverage": format_bool_yes_no,
                "flagged_high_dffits": format_bool_yes_no,
                "flagged_extreme_residual": format_bool_yes_no,
            },
            column_order=[
                "observation_index",
                "cooks_distance",
                "leverage",
                "dffits",
                "studentized_residual",
                "flagged_high_cook",
                "flagged_high_leverage",
                "flagged_high_dffits",
                "flagged_extreme_residual",
            ],
        ),
    },
)

# ==========================================================
# Guardrails
# ==========================================================

def evaluate_diagnostics_guardrails(run: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Guardrails for model diagnostics. Surfaces:
      - Multicollinearity (VIF tiers).
      - Heteroscedasticity (Breusch-Pagan).
      - Autocorrelation (Durbin-Watson).
      - Residual normality (Shapiro-Wilk / Jarque-Bera composite verdict).
      - Influence (Cook's distance, leverage, DFFITS, studentized residuals).
    """
    findings: list[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}

    max_vif = metrics.get("max_vif")
    bp_p = metrics.get("breusch_pagan_lm_p_value")
    hetero_flag = metrics.get("heteroscedasticity_flag_0_05")

    # Multicollinearity / VIF
    if max_vif is not None:
        try:
            v = float(max_vif)

            if v >= 10:
                findings.append(_new_finding(
                    category="multicollinearity",
                    severity="critical",
                    title="Severe multicollinearity possible",
                    message=(
                        "The maximum VIF is very high, suggesting severe multicollinearity. "
                        "Coefficient estimates and standard errors may be unstable."
                    ),
                    evidence={"max_vif": max_vif},
                    recommendation=(
                        "Inspect correlated predictors, remove redundant variables, "
                        "or use regularization/dimension reduction."
                    ),
                ))
            elif v >= 5:
                findings.append(_new_finding(
                    category="multicollinearity",
                    severity="warning",
                    title="Moderate multicollinearity possible",
                    message="The maximum VIF is elevated, suggesting possible multicollinearity.",
                    evidence={"max_vif": max_vif},
                    recommendation=(
                        "Inspect pairwise correlations and consider removing redundant predictors."
                    ),
                ))
            else:
                findings.append(_new_finding(
                    category="multicollinearity",
                    severity="info",
                    title="No apparent multicollinearity issue",
                    message="The maximum VIF does not indicate a multicollinearity problem.",
                    evidence={"max_vif": max_vif},
                ))
        except Exception:
            pass

    # Heteroscedasticity (BP)
    if bp_p is not None:
        try:
            p = float(bp_p)

            if p < 0.05 or hetero_flag is True:
                findings.append(_new_finding(
                    category="heteroscedasticity",
                    severity="warning",
                    title="Possible heteroscedasticity",
                    message=(
                        "The Breusch-Pagan test suggests non-constant error variance. "
                        "Classical OLS standard errors may be unreliable."
                    ),
                    evidence={
                        "breusch_pagan_lm_p_value": bp_p,
                        "heteroscedasticity_flag_0_05": hetero_flag,
                    },
                    recommendation=(
                        "Report HC3 robust standard errors (already computed in the linear "
                        "model), or consider transformation / model respecification."
                    ),
                ))
            else:
                findings.append(_new_finding(
                    category="heteroscedasticity",
                    severity="info",
                    title="No strong evidence of heteroscedasticity",
                    message=(
                        "The Breusch-Pagan test does not suggest strong evidence of heteroscedasticity."
                    ),
                    evidence={
                        "breusch_pagan_lm_p_value": bp_p,
                        "heteroscedasticity_flag_0_05": hetero_flag,
                    },
                ))
        except Exception:
            pass

    # Durbin-Watson / autocorrelation
    dw_stat = metrics.get("durbin_watson_statistic")
    dw_interp = metrics.get("durbin_watson_interpretation")

    if dw_stat is not None:
        try:
            if dw_interp in {
                "positive autocorrelation possible",
                "negative autocorrelation possible",
            }:
                findings.append(_new_finding(
                    category="autocorrelation",
                    severity="warning",
                    title="Possible residual autocorrelation",
                    message=(
                        f"The Durbin-Watson statistic ({dw_stat}) suggests possible "
                        f"{dw_interp}. Classical inference assumes independent residuals."
                    ),
                    evidence={
                        "durbin_watson_statistic": dw_stat,
                        "durbin_watson_interpretation": dw_interp,
                    },
                    recommendation=(
                        "If observations have a natural ordering (e.g., time series, panel data), "
                        "consider models that account for autocorrelation such as ARMA errors, "
                        "GLS, or HAC robust standard errors."
                    ),
                ))
            else:
                findings.append(_new_finding(
                    category="autocorrelation",
                    severity="info",
                    title="No strong residual autocorrelation",
                    message=(
                        "The Durbin-Watson statistic does not indicate strong residual autocorrelation."
                    ),
                    evidence={
                        "durbin_watson_statistic": dw_stat,
                        "durbin_watson_interpretation": dw_interp,
                    },
                ))
        except Exception:
            pass

    # Residual normality
    residuals_normal = metrics.get("residuals_appear_normal_at_0_05")

    if residuals_normal is False:
        findings.append(_new_finding(
            category="residual_distribution",
            severity="warning",
            title="Residuals deviate from normality",
            message=(
                "Shapiro-Wilk and/or Jarque-Bera reject the normality hypothesis at 0.05. "
                "Parametric confidence intervals and p-values may be approximate, especially "
                "for small samples."
            ),
            evidence={
                "shapiro_wilk_p_value": metrics.get("shapiro_wilk_p_value"),
                "jarque_bera_p_value": metrics.get("jarque_bera_p_value"),
            },
            recommendation=(
                "For small samples, consider bootstrap-based inference. For large samples, "
                "rely on the central limit theorem; the impact on inference is typically modest."
            ),
        ))
    elif residuals_normal is True:
        findings.append(_new_finding(
            category="residual_distribution",
            severity="info",
            title="Residuals consistent with normality",
            message=(
                "Shapiro-Wilk and/or Jarque-Bera do not reject normality of the residuals."
            ),
            evidence={
                "shapiro_wilk_p_value": metrics.get("shapiro_wilk_p_value"),
                "jarque_bera_p_value": metrics.get("jarque_bera_p_value"),
            },
        ))

    # Influence diagnostics
    n_high_cook = metrics.get("n_high_cooks_distance")
    n_high_leverage = metrics.get("n_high_leverage")
    n_high_dffits = metrics.get("n_high_dffits")
    n_extreme_resid = metrics.get("n_extreme_studentized_residual")

    influence_flagged = False

    influence_evidence = {
        "n_high_cooks_distance": n_high_cook,
        "n_high_leverage": n_high_leverage,
        "n_high_dffits": n_high_dffits,
        "n_extreme_studentized_residual": n_extreme_resid,
        "max_cooks_distance": metrics.get("max_cooks_distance"),
        "max_leverage": metrics.get("max_leverage"),
        "max_abs_dffits": metrics.get("max_abs_dffits"),
        "max_abs_studentized_residual": metrics.get("max_abs_studentized_residual"),
    }

    try:
        if n_high_cook is not None and int(n_high_cook) > 0:
            influence_flagged = True
            findings.append(_new_finding(
                category="influence",
                severity="warning",
                title=f"{n_high_cook} observation(s) with high Cook's distance",
                message=(
                    f"{n_high_cook} observation(s) exceed the Cook's distance threshold "
                    f"of 4/n. These rows have outsized influence on the coefficient estimates."
                ),
                evidence=influence_evidence,
                recommendation=(
                    "Inspect the flagged rows for data-entry errors or extreme but valid cases. "
                    "Consider refitting without them as a sensitivity check, and report whether "
                    "conclusions are robust."
                ),
            ))
    except Exception:
        pass

    try:
        if n_high_leverage is not None and int(n_high_leverage) > 0:
            influence_flagged = True
            findings.append(_new_finding(
                category="influence",
                severity="warning",
                title=f"{n_high_leverage} high-leverage observation(s)",
                message=(
                    f"{n_high_leverage} observation(s) have leverage above the rule-of-thumb "
                    f"threshold 2p/n. High-leverage points sit far from the predictor centroid "
                    f"and can disproportionately drive the fit."
                ),
                evidence=influence_evidence,
                recommendation=(
                    "Examine high-leverage rows. Even if not influential today, they may be "
                    "if the data changes; document them in your report."
                ),
            ))
    except Exception:
        pass

    try:
        if n_high_dffits is not None and int(n_high_dffits) > 0:
            influence_flagged = True
            findings.append(_new_finding(
                category="influence",
                severity="warning",
                title=f"{n_high_dffits} observation(s) with large |DFFITS|",
                message=(
                    f"{n_high_dffits} observation(s) exceed |DFFITS| > 2 * sqrt(p/n). These "
                    f"rows individually shift their own fitted values by an unusual amount."
                ),
                evidence=influence_evidence,
                recommendation=(
                    "Cross-reference with Cook's distance to identify globally influential rows."
                ),
            ))
    except Exception:
        pass

    try:
        if n_extreme_resid is not None and int(n_extreme_resid) > 0:
            influence_flagged = True
            findings.append(_new_finding(
                category="influence",
                severity="warning",
                title=f"{n_extreme_resid} extreme studentized residual(s)",
                message=(
                    f"{n_extreme_resid} observation(s) have externally studentized residuals "
                    f"with absolute value above 3. These are candidate outliers in the outcome."
                ),
                evidence=influence_evidence,
                recommendation=(
                    "Verify these rows; consider robust regression methods if outliers reflect "
                    "the real data-generating process rather than data errors."
                ),
            ))
    except Exception:
        pass

    if not influence_flagged and any(
        v is not None for v in [n_high_cook, n_high_leverage, n_high_dffits, n_extreme_resid]
    ):
        findings.append(_new_finding(
            category="influence",
            severity="info",
            title="No highly influential observations flagged",
            message=(
                "Cook's distance, leverage, DFFITS, and studentized residuals are all within "
                "rule-of-thumb thresholds."
            ),
            evidence=influence_evidence,
        ))

    return findings

PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="regression_diagnostics",
    display_name="Model Diagnostics",
    description=(
        "Run comprehensive regression diagnostics: VIF (multicollinearity), "
        "Breusch-Pagan (heteroscedasticity), Durbin-Watson (autocorrelation), "
        "Shapiro-Wilk and Jarque-Bera (residual normality), and Cook's distance, "
        "leverage, DFFITS, and studentized residuals (influence). "
        "When a previous regression model exists, this tool diagnoses that model "
        "using its stored model_spec instead of requiring manually supplied columns."
    ),
    usage_guidance=(
        "Prefer diagnosing the most recent regression_model analysis run. "
        "After run_multiple_regression has been executed, this tool may be called "
        "with source_analysis_run_id or with no target_col/feature_cols; it will "
        "resolve the latest regression model spec. If target_col/feature_cols are supplied "
        "but contain encoded coefficient terms such as region_North or segment_Corporate, "
        "the tool falls back to the stored model_spec and uses the original active-dataset "
        "features such as region and segment."
    ),
    evidence_categories=["regression_diagnostics", "model_diagnostics"],
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={
            "source_analysis_run_id": str,
            "target_col": str,
            "feature_cols": list,
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
))