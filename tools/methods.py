"""
methods.py

Tool-based statistical analysis methods for a data-analysis agent.

Design goals:
- Every registry tool returns a structured result: status = ok | warning | blocked | failed.
- No tool should return a bare error string.
- Regression, VIF diagnostics, and residual plots share one design-matrix preparation path.
- Plotting and analysis tools defend themselves against NaN / inf / common missing tokens.
- Risky data mutation tools remain requires_confirmation=True.

This file is intended as a drop-in replacement for the existing tools/methods.py style.
"""

import os
import uuid
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn is optional at runtime. Existing project imported it, so keep it when available.
try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from core.data_versions import create_child_data_version, make_audit_event
from tools.registry import registry


# ==========================================================
# Common structured result helpers
# ==========================================================

MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


def _ok(message: str, details: Optional[Dict[str, Any]] = None, artifacts: Optional[List[Dict[str, Any]]] = None, **extra):
    out = {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
    }
    if artifacts is not None:
        out["artifacts"] = artifacts
    out.update(extra)
    return out


def _warning(message: str, details: Optional[Dict[str, Any]] = None, artifacts: Optional[List[Dict[str, Any]]] = None, **extra):
    out = {
        "status": "warning",
        "message": message,
        "recoverable": False,
        "details": details or {},
    }
    if artifacts is not None:
        out["artifacts"] = artifacts
    out.update(extra)
    return out


def _blocked(error_code: str, message: str, details: Optional[Dict[str, Any]] = None,
             suggested_next_actions: Optional[List[str]] = None, recoverable: bool = True, **extra):
    out = {
        "status": "blocked",
        "error_code": error_code,
        "message": message,
        "recoverable": recoverable,
        "details": details or {},
    }
    if suggested_next_actions:
        out["suggested_next_actions"] = suggested_next_actions
    out.update(extra)
    return out


def _failed(error_code: str, message: str, exc: Optional[Exception] = None,
            details: Optional[Dict[str, Any]] = None, recoverable: bool = True, **extra):
    d = details.copy() if details else {}
    if exc is not None:
        d["exception_type"] = type(exc).__name__
        d["exception_message"] = str(exc)
    out = {
        "status": "failed",
        "error_code": error_code,
        "message": message,
        "recoverable": recoverable,
        "details": d,
    }
    out.update(extra)
    return out


def _json_safe_value(x: Any) -> Any:
    """Convert numpy/pandas scalar values to JSON-safe Python values."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def _round_or_none(x: Any, digits: int = 6) -> Optional[float]:
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
        # Some context implementations may not accept a default argument.
        try:
            val = context.get_arg(name)
            return default if val is None else val
        except Exception:
            return default
    except Exception:
        return default


def _load_df(context) -> pd.DataFrame:
    df = context.load_df()
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("context.load_df() did not return a valid pandas DataFrame.")
    return df.copy()


def _save_df(context, df: pd.DataFrame) -> None:
    """Save DataFrame using context.save_df if available; otherwise parquet to context.file_path."""
    if hasattr(context, "save_df") and callable(getattr(context, "save_df")):
        context.save_df(df)
        return
    if hasattr(context, "file_path") and context.file_path:
        df.to_parquet(context.file_path, index=False)
        return
    raise ValueError("No save target found. Expected context.save_df(df) or context.file_path.")


def _select_columns(df: pd.DataFrame, cols: Any) -> List[str]:
    if cols is None or cols == "all":
        return df.columns.tolist()
    if isinstance(cols, str):
        cols = [cols]
    if not isinstance(cols, list):
        raise ValueError("columns must be 'all', a column name, or a list of column names.")
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")
    return cols


def _normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
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
    return df


def _replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().replace([np.inf, -np.inf], np.nan)


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _replace_inf(_normalize_missing_tokens(df))


def _count_inf(df: pd.DataFrame) -> int:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return 0
    try:
        return int(np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum())
    except Exception:
        return 0


def _safe_mode(s: pd.Series) -> Any:
    m = s.dropna().mode()
    return np.nan if len(m) == 0 else m.iloc[0]


def _infer_column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def _try_convert_numeric_object_columns(df: pd.DataFrame, cols: List[str], threshold: float = 0.85) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    df = df.copy()
    conversions = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        non_missing = s.dropna()
        if len(non_missing) == 0:
            continue
        coerced = pd.to_numeric(non_missing, errors="coerce")
        numeric_rate = float(coerced.notna().mean())
        if numeric_rate >= threshold:
            before_dtype = str(df[col].dtype)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            conversions.append({
                "column": col,
                "from_dtype": before_dtype,
                "to_dtype": str(df[col].dtype),
                "numeric_parse_rate": numeric_rate,
            })
    return df, conversions


def _is_id_like(col: str, s: pd.Series) -> bool:
    name = str(col).lower().strip()
    if name in {"id", "index", "row", "record", "case", "subject_id", "student_id"}:
        return True
    if name.endswith("_id") or name.endswith("id"):
        return True
    non_missing = s.dropna()
    if len(non_missing) == 0:
        return False
    # High uniqueness ratio often indicates identifier-like columns.
    unique_ratio = non_missing.nunique() / max(len(non_missing), 1)
    return bool(unique_ratio > 0.95 and len(non_missing) > 20)


# ==========================================================
# Shared regression design-matrix preparation
# ==========================================================

def prepare_regression_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    *,
    max_missing_rate: float = 0.40,
    max_categorical_levels: int = 10,
    numeric_parse_threshold: float = 0.85,
    min_n_per_parameter: int = 3,
    drop_id_like: bool = True,
    allow_categorical: bool = True,
) -> Dict[str, Any]:
    """
    Prepare y and X for OLS/VIF/residual plotting.

    This function intentionally returns y and X as pandas objects for internal tool use.
    Do not expose those objects directly to the LLM; expose details only.
    """
    if not target_col:
        return _blocked(
            "MISSING_TARGET",
            "target_col is required for regression.",
            suggested_next_actions=["Ask the user to specify an outcome variable."],
        )
    if not feature_cols or not isinstance(feature_cols, list):
        return _blocked(
            "MISSING_FEATURES",
            "feature_cols must be a non-empty list.",
            suggested_next_actions=["Ask the user to specify predictors or run candidate selection."],
        )

    df = _standardize_dataframe(df)
    all_cols = [target_col] + list(feature_cols)
    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        return _blocked(
            "COLUMNS_NOT_FOUND",
            f"Columns not found: {missing_cols}",
            details={"missing_cols": missing_cols},
            suggested_next_actions=["Inspect dataset columns and retry with valid column names."],
        )

    work = df[all_cols].copy()

    y = pd.to_numeric(work[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    target_nonmissing = int(y.notna().sum())
    if target_nonmissing < 3:
        return _blocked(
            "TARGET_NOT_NUMERIC_OR_TOO_MISSING",
            f"Target column '{target_col}' cannot be used as a numeric regression outcome.",
            details={"target": target_col, "target_non_missing_after_numeric_conversion": target_nonmissing},
            suggested_next_actions=["Choose a numeric outcome or inspect target column coding."],
        )

    X_parts: List[pd.DataFrame] = []
    used_features: List[Dict[str, Any]] = []
    excluded_features: List[Dict[str, Any]] = []

    for col in feature_cols:
        s = work[col]
        missing_rate = float(s.isna().mean())
        non_missing = s.dropna()

        if drop_id_like and _is_id_like(col, s):
            excluded_features.append({"column": col, "reason": "id_like", "missing_rate": missing_rate})
            continue
        if missing_rate > max_missing_rate:
            excluded_features.append({"column": col, "reason": "high_missing_rate", "missing_rate": missing_rate})
            continue
        if non_missing.nunique() <= 1:
            excluded_features.append({"column": col, "reason": "constant_or_all_missing", "unique_non_missing": int(non_missing.nunique())})
            continue

        if pd.api.types.is_bool_dtype(s):
            numeric_s = s.astype(float)
            X_parts.append(pd.DataFrame({col: numeric_s}, index=work.index))
            used_features.append({"column": col, "type": "boolean_as_numeric", "encoded_columns": [col]})
            continue

        if pd.api.types.is_numeric_dtype(s):
            numeric_s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_parts.append(pd.DataFrame({col: numeric_s}, index=work.index))
            used_features.append({"column": col, "type": "numeric", "encoded_columns": [col]})
            continue

        # Object/string/category: numeric-like or categorical.
        numeric_candidate = pd.to_numeric(non_missing, errors="coerce")
        numeric_rate = float(numeric_candidate.notna().mean()) if len(non_missing) else 0.0
        if numeric_rate >= numeric_parse_threshold:
            numeric_s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_parts.append(pd.DataFrame({col: numeric_s}, index=work.index))
            used_features.append({
                "column": col,
                "type": "numeric_like_object",
                "numeric_parse_rate": numeric_rate,
                "encoded_columns": [col],
            })
            continue

        if not allow_categorical:
            excluded_features.append({"column": col, "reason": "categorical_not_allowed", "missing_rate": missing_rate})
            continue

        n_levels = int(non_missing.nunique())
        if n_levels > max_categorical_levels:
            excluded_features.append({"column": col, "reason": "too_many_categorical_levels", "n_levels": n_levels, "missing_rate": missing_rate})
            continue

        dummies = pd.get_dummies(s, prefix=str(col), drop_first=True, dummy_na=False, dtype=float)
        dummies = dummies.reindex(index=work.index)
        if dummies.shape[1] == 0:
            excluded_features.append({"column": col, "reason": "categorical_encoding_produced_no_columns", "n_levels": n_levels})
            continue
        X_parts.append(dummies)
        used_features.append({"column": col, "type": "categorical_encoded", "n_levels": n_levels, "encoded_columns": dummies.columns.tolist()})

    if not X_parts:
        return _blocked(
            "NO_USABLE_PREDICTORS",
            "No usable predictors remain after preprocessing.",
            details={"target": target_col, "excluded_features": excluded_features},
            suggested_next_actions=["Select fewer predictors, include numeric predictors, or lower missingness/cardinality constraints carefully."],
        )

    X = pd.concat(X_parts, axis=1)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Drop duplicate columns if dummy prefixes collide.
    X = X.loc[:, ~X.columns.duplicated()].copy()

    design = pd.concat([y.rename(target_col), X], axis=1)
    design = design.replace([np.inf, -np.inf], np.nan).dropna()

    y_clean = design[target_col].astype(float)
    X_clean = design.drop(columns=[target_col]).astype(float)

    # Drop zero variance encoded columns after complete-case filtering.
    zero_var_cols = [c for c in X_clean.columns if X_clean[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        X_clean = X_clean.drop(columns=zero_var_cols)
        for f in used_features:
            f["encoded_columns"] = [c for c in f.get("encoded_columns", []) if c not in zero_var_cols]
        excluded_features.extend([{"column": c, "reason": "zero_variance_after_complete_case_filtering"} for c in zero_var_cols])

    if X_clean.shape[1] == 0:
        return _blocked(
            "NO_USABLE_PREDICTORS_AFTER_FILTERING",
            "No usable predictors remain after complete-case filtering.",
            details={"target": target_col, "excluded_features": excluded_features},
            suggested_next_actions=["Select fewer predictors or inspect missingness."],
        )

    n_eff = int(len(y_clean))
    p_eff = int(X_clean.shape[1])
    min_required = max(p_eff + 2, int(min_n_per_parameter) * (p_eff + 1))

    details = {
        "target": target_col,
        "n_eff": n_eff,
        "p_eff": p_eff,
        "min_required": int(min_required),
        "raw_feature_count": int(len(feature_cols)),
        "encoded_column_count": p_eff,
        "used_features": used_features,
        "excluded_features": excluded_features,
        "encoded_columns": X_clean.columns.tolist(),
        "max_missing_rate": max_missing_rate,
        "max_categorical_levels": max_categorical_levels,
        "min_n_per_parameter": min_n_per_parameter,
    }

    if n_eff < min_required:
        return _blocked(
            "INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
            f"Insufficient effective sample size: only {n_eff} rows after preprocessing, "
            f"but the design matrix has {p_eff} predictors; at least {min_required} rows are recommended.",
            details=details,
            suggested_next_actions=[
                "reduce_predictors",
                "run_select_regression_candidates",
                "use_numeric_low_missing_predictors_first",
                "ask_user_to_choose_a_smaller_model",
                "consider_imputation_after_user_confirmation",
            ],
        )

    return {
        "status": "ok",
        "message": "Regression design matrix prepared successfully.",
        "recoverable": False,
        "y": y_clean,
        "X": X_clean,
        "details": details,
    }


# ==========================================================
# Dataset inspection and summary tools
# ==========================================================

@registry.register()
def inspect_dataset(context):
    """Inspect dataset shape, column types, missingness, infinity counts, and simple column profiles."""
    try:
        df = _load_df(context)
        std = _standardize_dataframe(df)
        columns = []
        for col in std.columns:
            s = std[col]
            columns.append({
                "name": str(col),
                "dtype": str(s.dtype),
                "kind": _infer_column_kind(s),
                "missing_count": int(s.isna().sum()),
                "missing_rate": round(float(s.isna().mean()), 6),
                "unique_count": int(s.nunique(dropna=True)),
                "id_like": bool(_is_id_like(col, s)),
            })
        return _ok("Dataset inspection completed.", {
            "shape": {"rows": int(std.shape[0]), "columns": int(std.shape[1])},
            "total_missing": int(std.isna().sum().sum()),
            "total_inf": _count_inf(std),
            "columns": columns,
        })
    except Exception as e:
        return _failed("INSPECT_DATASET_EXCEPTION", "Dataset inspection failed.", e)


@registry.register()
def get_summary_stats(context):
    """Get descriptive summary for numeric and categorical columns."""
    try:
        df = _standardize_dataframe(_load_df(context))
        numeric = df.select_dtypes(include=[np.number])
        categorical = df.select_dtypes(exclude=[np.number])
        details: Dict[str, Any] = {}
        if numeric.shape[1] > 0:
            details["numeric_summary"] = numeric.describe().replace([np.inf, -np.inf], np.nan).round(6).where(pd.notna(numeric.describe()), None).to_dict()
        else:
            details["numeric_summary"] = {}
        cat_summary = {}
        for col in categorical.columns:
            vc = categorical[col].value_counts(dropna=True).head(10)
            cat_summary[str(col)] = {
                "missing_count": int(categorical[col].isna().sum()),
                "unique_count": int(categorical[col].nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }
        details["categorical_summary"] = cat_summary
        return _ok("Summary statistics completed.", details)
    except Exception as e:
        return _failed("SUMMARY_STATS_EXCEPTION", "Summary statistics failed.", e)


@registry.register()
def summarize_columns(context):
    """Summarize selected columns. Args: columns='all' or list[str]."""
    try:
        df = _standardize_dataframe(_load_df(context))
        cols = _select_columns(df, _get_arg(context, "columns", "all"))
        summary = {}
        for col in cols:
            s = df[col]
            item = {
                "dtype": str(s.dtype),
                "kind": _infer_column_kind(s),
                "missing_count": int(s.isna().sum()),
                "missing_rate": round(float(s.isna().mean()), 6),
                "unique_count": int(s.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(s):
                clean = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                item.update({
                    "n": int(len(clean)),
                    "mean": _round_or_none(clean.mean()),
                    "std": _round_or_none(clean.std()),
                    "min": _round_or_none(clean.min()),
                    "median": _round_or_none(clean.median()),
                    "max": _round_or_none(clean.max()),
                })
            else:
                item["top_values"] = {str(k): int(v) for k, v in s.value_counts(dropna=True).head(10).items()}
            summary[str(col)] = item
        return _ok("Column summary completed.", {"summary": summary})
    except Exception as e:
        return _failed("SUMMARIZE_COLUMNS_EXCEPTION", "Column summary failed.", e)


@registry.register()
def missingness_report(context):
    """Report missingness and non-finite values by column. Args: columns='all' or list[str]."""
    try:
        df = _standardize_dataframe(_load_df(context))
        cols = _select_columns(df, _get_arg(context, "columns", "all"))
        rows = []
        for col in cols:
            s = df[col]
            inf_count = 0
            if pd.api.types.is_numeric_dtype(s):
                inf_count = int(np.isinf(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)).sum())
            rows.append({
                "column": str(col),
                "dtype": str(s.dtype),
                "missing_count": int(s.isna().sum()),
                "missing_rate": round(float(s.isna().mean()), 6),
                "inf_count": inf_count,
                "unique_count": int(s.nunique(dropna=True)),
            })
        rows.sort(key=lambda r: (r["missing_rate"], r["inf_count"]), reverse=True)
        return _ok("Missingness report completed.", {"columns": rows, "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])}})
    except Exception as e:
        return _failed("MISSINGNESS_REPORT_EXCEPTION", "Missingness report failed.", e)


# ==========================================================
# Data cleaning tool
# ==========================================================

@registry.register(requires_confirmation=True)
def clean_data(context):
    """
    Robust data cleaning tool.

    Args:
    - action_type: 'standardize_missing' | 'impute' | 'drop' | 'plot_safe'
    - columns: 'all' or list[str]
    - strategy:
        impute: 'mean' | 'median' | 'mode' | 'interpolate' | 'constant'
        drop: 'rows' | 'cols'
    Optional:
    - missing_col_threshold: default 0.4
    - constant_value
    - numeric_parse_threshold: default 0.85
    - save: default True
    """
    try:
        df = _load_df(context)
        action = _get_arg(context, "action_type", "standardize_missing")
        cols_arg = _get_arg(context, "columns", "all")
        strategy = _get_arg(context, "strategy", "median")

        workspace_dir = getattr(context, "workspace_dir", None) or _get_arg(context, "workspace_dir", None) or "."

        # data_versions = getattr(context, "data_versions", None) or _get_arg(context, "data_versions", []) or []
        active_data_version_id = (
                getattr(context, "active_data_version_id", None)
                or _get_arg(context, "active_data_version_id", None)
        )

        parent_version_id = active_data_version_id or "unknown"

        missing_col_threshold = float(_get_arg(context, "missing_col_threshold", 0.4))
        constant_value = _get_arg(context, "constant_value", None)
        numeric_parse_threshold = float(_get_arg(context, "numeric_parse_threshold", 0.85))
        save = bool(_get_arg(context, "save", True))

        original_shape = df.shape
        original_missing = int(df.isna().sum().sum())
        original_inf = _count_inf(df)
        cols = _select_columns(df, cols_arg)

        audit = {
            "action_type": action,
            "strategy": strategy,
            "requested_columns": cols_arg,
            "resolved_columns": cols,
            "original_shape": original_shape,
            "original_missing": original_missing,
            "original_inf": original_inf,
            "numeric_conversions": [],
            "imputations": [],
            "dropped_rows": 0,
            "dropped_columns": [],
            "plot_safe_rows_removed": 0,
            "warnings": [],
        }

        df = _standardize_dataframe(df)
        df, conversions = _try_convert_numeric_object_columns(df, cols, threshold=numeric_parse_threshold)
        audit["numeric_conversions"] = conversions
        cols = [c for c in cols if c in df.columns]

        if action == "standardize_missing":
            pass

        elif action == "impute":
            for col in cols:
                s = df[col]
                before_missing = int(s.isna().sum())
                if before_missing == 0:
                    continue
                kind = _infer_column_kind(s)
                if kind == "numeric":
                    if s.dropna().empty:
                        fill_value = 0 if constant_value is None else constant_value
                    elif strategy == "mean":
                        fill_value = s.mean()
                    elif strategy == "median":
                        fill_value = s.median()
                    elif strategy == "mode":
                        fill_value = _safe_mode(s)
                        if pd.isna(fill_value):
                            fill_value = 0
                    elif strategy == "constant":
                        fill_value = 0 if constant_value is None else constant_value
                    elif strategy == "interpolate":
                        out = s.interpolate(method="linear", limit_direction="both")
                        if out.isna().any():
                            out = out.fillna(s.median() if not s.dropna().empty else 0)
                        df[col] = out
                        fill_value = "linear_interpolation_with_fallback"
                    else:
                        return _blocked("UNSUPPORTED_IMPUTE_STRATEGY", f"Unsupported imputation strategy: {strategy}", suggested_next_actions=["Use mean, median, mode, interpolate, or constant."])
                    if strategy != "interpolate":
                        df[col] = s.fillna(fill_value)
                elif kind == "datetime":
                    fill_value = _safe_mode(s) if strategy == "mode" else None
                    if fill_value is None or pd.isna(fill_value):
                        fill_value = s.dropna().iloc[0] if not s.dropna().empty else pd.NaT
                    df[col] = s.fillna(fill_value)
                else:
                    fill_value = _safe_mode(s)
                    if pd.isna(fill_value) or strategy == "constant":
                        fill_value = "Unknown" if constant_value is None else constant_value
                    df[col] = s.fillna(fill_value)

                audit["imputations"].append(
                    {
                    "column": col,
                    "kind": kind,
                    "strategy_requested": strategy,
                    "fill_value": str(_json_safe_value(fill_value)),
                    "missing_before": before_missing,
                    "missing_after": int(df[col].isna().sum()),
                }
                )

        elif action == "drop":
            if strategy == "rows":
                before = len(df)
                df = df.dropna(subset=cols)
                audit["dropped_rows"] = int(before - len(df))
            elif strategy == "cols":
                missing_rates = df[cols].isna().mean()
                to_drop = missing_rates[missing_rates > missing_col_threshold].index.tolist()
                df = df.drop(columns=to_drop)
                audit["dropped_columns"] = to_drop
            else:
                return _blocked("UNSUPPORTED_DROP_STRATEGY", f"Unsupported drop strategy: {strategy}", suggested_next_actions=["Use rows or cols."])

        elif action == "plot_safe":
            existing = [c for c in cols if c in df.columns]
            numeric_cols = [c for c in existing if pd.api.types.is_numeric_dtype(df[c])]
            before = len(df)
            if numeric_cols:
                df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
                finite_mask = np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all(axis=1)
                df = df.loc[finite_mask].copy()
            audit["plot_safe_rows_removed"] = int(before - len(df))

        else:
            return _blocked(
                "UNSUPPORTED_CLEANING_ACTION",
                f"Unsupported action_type: {action}",
                suggested_next_actions=["Use standardize_missing, impute, drop, or plot_safe."],
            )

        df = _replace_inf(df)

        data_version_update = None

        if save:
            new_version = create_child_data_version(
                df=df,
                workspace_dir=workspace_dir,
                parent_version_id=parent_version_id,
                operation=f"clean_data:{action}-{strategy}",
                created_by="clean_data",
                description=f"Data cleaned using action_type={action}, strategy={strategy}.",
                metadata={
                    "action_type": action,
                    "strategy": strategy,
                    "requested_columns": cols_arg,
                    "resolved_columns": cols,
                    "audit": audit,
                },
            )

            audit_event = make_audit_event(
                event_type="data_cleaned",
                version_id=new_version["version_id"],
                parent_version_id=parent_version_id,
                tool_name="clean_data",
                description=f"Created new data version {new_version['version_id']} from {parent_version_id}.",
                details={
                    "action_type": action,
                    "strategy": strategy,
                    "old_shape": tuple(original_shape),
                    "new_shape": tuple(df.shape),
                    "dropped_rows": audit.get("dropped_rows", 0),
                    "dropped_columns": audit.get("dropped_columns", []),
                    "imputations": audit.get("imputations", []),
                },
            )

            data_version_update = {
                "new_version": new_version,
                "active_data_version_id": new_version["version_id"],
                "audit_event": audit_event,
            }

        selected_cols = [c for c in cols if c in df.columns]
        selected_missing_after = {c: int(df[c].isna().sum()) for c in selected_cols}
        selected_inf_after = {c: int(np.isinf(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)).sum()) if pd.api.types.is_numeric_dtype(df[c]) else 0 for c in selected_cols}
        total_missing_after = int(df.isna().sum().sum())
        total_inf_after = _count_inf(df)

        details = {
            "action_type": action,
            "strategy": strategy,
            "original_shape": original_shape,
            "final_shape": df.shape,
            "total_missing_before": original_missing,
            "total_missing_after": total_missing_after,
            "total_inf_before": original_inf,
            "total_inf_after": total_inf_after,
            "selected_missing_after": selected_missing_after,
            "selected_inf_after": selected_inf_after,
            "final_columns": df.columns.tolist(),
        }

        if data_version_update:
            details["old_version_id"] = parent_version_id
            details["new_version_id"] = data_version_update["active_data_version_id"]
            details["data_version_created"] = True
        else:
            details["old_version_id"] = parent_version_id
            details["new_version_id"] = None
            details["data_version_created"] = False

        if len(df) == 0:
            return _warning(
                "Data cleaning completed, but the resulting dataset has 0 rows.",
                details=details,
                audit=audit,
                data_version_update=data_version_update,
            )

        if total_inf_after > 0 or (
                action in {"impute", "plot_safe"} and any(v > 0 for v in selected_missing_after.values())):
            return _warning(
                "Data cleaning completed with remaining missing/non-finite values.",
                details=details,
                audit=audit,
                data_version_update=data_version_update,
            )

        return _ok(
            "Data cleaning completed and a new data version was created." if data_version_update else "Data cleaning completed.",
            details=details,
            audit=audit,
            data_version_update=data_version_update,
        )



    except Exception as e:
        return _failed(
            "CLEAN_DATA_EXCEPTION",
            "Data cleaning failed.",
            e,
            suggested_next_actions=["Run inspect_dataset first.", "Check selected columns.", "Try action_type='standardize_missing'."],
        )


# ==========================================================
# Correlation, tests, ANOVA, chi-square
# ==========================================================

@registry.register()
def get_correlation_matrix(context):
    """Compute Pearson correlation matrix for numeric variables. Args: columns='all' optional."""
    try:
        df = _standardize_dataframe(_load_df(context))
        cols_arg = _get_arg(context, "columns", "all")
        cols = _select_columns(df, cols_arg)
        numeric_df = df[cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.dropna(axis=1, how="all")
        if numeric_df.shape[1] < 2:
            return _blocked("INSUFFICIENT_NUMERIC_COLUMNS", "At least two numeric columns are required for correlation.", details={"numeric_columns": numeric_df.columns.tolist()})
        corr = numeric_df.corr().round(6)
        return _ok("Correlation matrix completed.", {"method": "pearson", "columns": corr.columns.tolist(), "correlation_matrix": corr.to_dict()})
    except Exception as e:
        return _failed("CORRELATION_EXCEPTION", "Correlation matrix failed.", e)


@registry.register()
def run_independent_t_test(context):
    """Welch independent t-test. Args: target_col, group_col, group1_val, group2_val."""
    try:
        df = _standardize_dataframe(_load_df(context))
        target_col = _get_arg(context, "target_col")
        group_col = _get_arg(context, "group_col")
        group1_val = _get_arg(context, "group1_val")
        group2_val = _get_arg(context, "group2_val")
        if not all([target_col, group_col, group1_val is not None, group2_val is not None]):
            return _blocked("MISSING_T_TEST_ARGS", "target_col, group_col, group1_val, and group2_val are required.")
        if target_col not in df.columns or group_col not in df.columns:
            return _blocked("COLUMNS_NOT_FOUND", "target_col or group_col not found.", details={"target_col": target_col, "group_col": group_col})
        y = pd.to_numeric(df[target_col], errors="coerce")
        g1 = y[df[group_col] == group1_val].replace([np.inf, -np.inf], np.nan).dropna()
        g2 = y[df[group_col] == group2_val].replace([np.inf, -np.inf], np.nan).dropna()
        if len(g1) < 2 or len(g2) < 2:
            return _blocked("INSUFFICIENT_GROUP_SIZE", "Each group needs at least 2 valid numeric observations.", details={"group1_n": int(len(g1)), "group2_n": int(len(g2))})
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
        return _ok("Welch independent t-test completed.", {
            "method": "Welch two-sample t-test",
            "group1": str(group1_val), "group1_n": int(len(g1)), "group1_mean": _round_or_none(g1.mean()),
            "group2": str(group2_val), "group2_n": int(len(g2)), "group2_mean": _round_or_none(g2.mean()),
            "t_statistic": _round_or_none(t_stat), "p_value": _round_or_none(p_val),
            "significant_at_0_05": bool(p_val < 0.05) if math.isfinite(float(p_val)) else None,
        })
    except Exception as e:
        return _failed("T_TEST_EXCEPTION", "T-test failed.", e)


@registry.register()
def run_anova(context):
    """One-way ANOVA. Args: target_col numeric, group_col categorical."""
    try:
        df = _standardize_dataframe(_load_df(context))
        target_col = _get_arg(context, "target_col")
        group_col = _get_arg(context, "group_col")
        if not target_col or not group_col:
            return _blocked("MISSING_ANOVA_ARGS", "target_col and group_col are required.")
        if target_col not in df.columns or group_col not in df.columns:
            return _blocked("COLUMNS_NOT_FOUND", "target_col or group_col not found.", details={"target_col": target_col, "group_col": group_col})
        work = pd.DataFrame({"y": pd.to_numeric(df[target_col], errors="coerce"), "g": df[group_col]}).replace([np.inf, -np.inf], np.nan).dropna()
        groups = [v.to_numpy(dtype=float) for _, v in work.groupby("g")["y"] if len(v) >= 2]
        if len(groups) < 2:
            return _blocked("INSUFFICIENT_GROUPS", "ANOVA requires at least two groups with at least two observations each.", details={"valid_group_count": len(groups)})
        f_stat, p_val = stats.f_oneway(*groups)
        return _ok("One-way ANOVA completed.", {"F_statistic": _round_or_none(f_stat), "p_value": _round_or_none(p_val), "valid_group_count": len(groups), "significant_at_0_05": bool(p_val < 0.05)})
    except Exception as e:
        return _failed("ANOVA_EXCEPTION", "ANOVA failed.", e)


@registry.register()
def run_chi_square(context):
    """Chi-square test of independence. Args: col1, col2."""
    try:
        df = _standardize_dataframe(_load_df(context))
        col1 = _get_arg(context, "col1")
        col2 = _get_arg(context, "col2")
        if not col1 or not col2:
            return _blocked("MISSING_CHI_SQUARE_ARGS", "col1 and col2 are required.")
        if col1 not in df.columns or col2 not in df.columns:
            return _blocked("COLUMNS_NOT_FOUND", "col1 or col2 not found.", details={"col1": col1, "col2": col2})
        table = pd.crosstab(df[col1], df[col2])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return _blocked("INSUFFICIENT_CONTINGENCY_TABLE", "Chi-square test requires at least a 2x2 contingency table.", details={"shape": table.shape})
        chi2, p, dof, expected = chi2_contingency(table)
        return _ok("Chi-square test completed.", {"chi2_statistic": _round_or_none(chi2), "p_value": _round_or_none(p), "degrees_of_freedom": int(dof), "table_shape": table.shape, "significant_at_0_05": bool(p < 0.05)})
    except Exception as e:
        return _failed("CHI_SQUARE_EXCEPTION", "Chi-square test failed.", e)


# ==========================================================
# Regression tools
# ==========================================================

@registry.register()
def select_regression_candidates(context):
    """
    Select a conservative baseline feature set for regression.
    Args: target_col, candidate_cols optional, max_features default 8.
    """
    try:
        df = _standardize_dataframe(_load_df(context))
        target_col = _get_arg(context, "target_col")
        candidate_cols = _get_arg(context, "candidate_cols", None)
        max_features = int(_get_arg(context, "max_features", 8))
        max_missing_rate = float(_get_arg(context, "max_missing_rate", 0.25))
        max_categorical_levels = int(_get_arg(context, "max_categorical_levels", 5))

        if not target_col or target_col not in df.columns:
            return _blocked("TARGET_NOT_FOUND", "A valid target_col is required.", details={"target_col": target_col})
        if candidate_cols is None or candidate_cols == "all":
            candidate_cols = [c for c in df.columns if c != target_col]
        else:
            candidate_cols = _select_columns(df, candidate_cols)
            candidate_cols = [c for c in candidate_cols if c != target_col]

        selected, excluded = [], []
        for col in candidate_cols:
            s = df[col]
            missing_rate = float(s.isna().mean())
            if _is_id_like(col, s):
                excluded.append({"column": col, "reason": "id_like"})
                continue
            if missing_rate > max_missing_rate:
                excluded.append({"column": col, "reason": "high_missing_rate", "missing_rate": missing_rate})
                continue
            if s.dropna().nunique() <= 1:
                excluded.append({"column": col, "reason": "constant_or_all_missing"})
                continue
            if pd.api.types.is_numeric_dtype(s):
                selected.append({"column": col, "type": "numeric", "missing_rate": missing_rate, "score": 1.0 - missing_rate})
                continue
            # numeric-like object
            non_missing = s.dropna()
            numeric_rate = float(pd.to_numeric(non_missing, errors="coerce").notna().mean()) if len(non_missing) else 0.0
            if numeric_rate >= 0.85:
                selected.append({"column": col, "type": "numeric_like_object", "missing_rate": missing_rate, "score": 0.9 - missing_rate})
                continue
            n_levels = int(non_missing.nunique())
            if n_levels <= max_categorical_levels:
                selected.append({"column": col, "type": "low_cardinality_categorical", "n_levels": n_levels, "missing_rate": missing_rate, "score": 0.6 - missing_rate - 0.02 * n_levels})
            else:
                excluded.append({"column": col, "reason": "too_many_categorical_levels", "n_levels": n_levels})

        selected = sorted(selected, key=lambda x: x.get("score", 0), reverse=True)[:max_features]
        selected_cols = [x["column"] for x in selected]
        if not selected_cols:
            return _blocked("NO_CANDIDATE_FEATURES", "No conservative baseline predictors found.", details={"excluded": excluded})
        return _ok("Regression candidate selection completed.", {"target": target_col, "selected_features": selected_cols, "selected_feature_details": selected, "excluded_features": excluded})
    except Exception as e:
        return _failed("SELECT_REGRESSION_CANDIDATES_EXCEPTION", "Candidate selection failed.", e)


@registry.register()
def preflight_regression_design(context):
    """Check if a regression design is estimable before fitting. Args: target_col, feature_cols."""
    try:
        df = _load_df(context)
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
        return _ok("Regression design preflight passed.", prep["details"])
    except Exception as e:
        return _failed("PREFLIGHT_REGRESSION_EXCEPTION", "Regression preflight failed.", e)


@registry.register()
def run_multiple_regression(context):
    """Run OLS multiple regression. Args: target_col, feature_cols."""
    try:
        df = _load_df(context)
        target_col = _get_arg(context, "target_col")
        feature_cols = _get_arg(context, "feature_cols", [])
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
        y, X = prep["y"], prep["X"]
        X_const = sm.add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X_const).fit()

        conf = model.conf_int()
        coef_table = []
        for term in model.params.index:
            coef_table.append({
                "term": str(term),
                "coef": _round_or_none(model.params[term]),
                "std_err": _round_or_none(model.bse[term]),
                "t": _round_or_none(model.tvalues[term]),
                "p_value": _round_or_none(model.pvalues[term]),
                "ci_lower": _round_or_none(conf.loc[term, 0]) if term in conf.index else None,
                "ci_upper": _round_or_none(conf.loc[term, 1]) if term in conf.index else None,
            })
        details = {
            **prep["details"],
            "model_type": "OLS multiple linear regression",
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
        }
        status = "ok"
        msg = "Multiple regression completed successfully."
        if int(model.nobs) < 30 or prep["details"]["n_eff"] < 5 * (prep["details"]["p_eff"] + 1):
            status = "warning"
            msg = "Multiple regression completed, but sample size is small relative to the number of predictors. Interpret cautiously."
        return _warning(msg, details) if status == "warning" else _ok(msg, details)
    except Exception as e:
        return _failed("OLS_FIT_EXCEPTION", "OLS model fitting failed.", e)


@registry.register(requires_confirmation=False)
def regression_diagnostics(context):
    """Run VIF and Breusch-Pagan diagnostics using the same prepared design matrix as OLS. Args: target_col, feature_cols."""
    try:
        df = _load_df(context)
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
        y, X = prep["y"], prep["X"]
        X_const = sm.add_constant(X, has_constant="add")
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
            vif_rows.append({"term": str(col), "vif": vif_value, "flag": bool(vif_value is not None and vif_value > 10)})

        bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = sms.het_breuschpagan(model.resid, model.model.exog)
        details = {
            **prep["details"],
            "vif": vif_rows,
            "breusch_pagan": {
                "lm_statistic": _round_or_none(bp_stat),
                "lm_p_value": _round_or_none(bp_pvalue),
                "f_statistic": _round_or_none(bp_fstat),
                "f_p_value": _round_or_none(bp_fpvalue),
                "heteroscedasticity_flag_0_05": bool(bp_pvalue < 0.05) if math.isfinite(float(bp_pvalue)) else None,
            },
        }
        msg = "Regression diagnostics completed successfully."
        if any(r["flag"] for r in vif_rows) or (details["breusch_pagan"]["heteroscedasticity_flag_0_05"] is True):
            return _warning("Regression diagnostics completed with statistical warnings.", details)
        return _ok(msg, details)
    except Exception as e:
        return _failed("REGRESSION_DIAGNOSTICS_EXCEPTION", "Regression diagnostics failed.", e)


@registry.register()
def run_logistic_regression(context):
    """Run binary logistic regression. Args: target_col, feature_cols."""
    try:
        df = _standardize_dataframe(_load_df(context))
        target_col = _get_arg(context, "target_col")
        feature_cols = _get_arg(context, "feature_cols", [])
        if not target_col or target_col not in df.columns:
            return _blocked("TARGET_NOT_FOUND", "A valid target_col is required.", details={"target_col": target_col})
        y_raw = df[target_col]
        # Accept numeric 0/1 or two-level categorical.
        y_num = pd.to_numeric(y_raw, errors="coerce")
        if y_num.dropna().nunique() == 2:
            vals = sorted(y_num.dropna().unique().tolist())
            y = y_num.map({vals[0]: 0, vals[1]: 1})
        else:
            levels = y_raw.dropna().unique().tolist()
            if len(levels) != 2:
                return _blocked("TARGET_NOT_BINARY", "Logistic regression requires a binary target.", details={"target": target_col, "levels": [str(x) for x in levels[:20]]})
            y = y_raw.map({levels[0]: 0, levels[1]: 1}).astype(float)

        temp = df.copy()
        temp[target_col] = y
        prep = prepare_regression_data(
            temp,
            target_col,
            feature_cols,
            max_missing_rate=float(_get_arg(context, "max_missing_rate", 0.40)),
            max_categorical_levels=int(_get_arg(context, "max_categorical_levels", 10)),
            numeric_parse_threshold=float(_get_arg(context, "numeric_parse_threshold", 0.85)),
            min_n_per_parameter=int(_get_arg(context, "min_n_per_parameter", 5)),
        )
        if prep.get("status") != "ok":
            return prep
        y_clean, X = prep["y"], prep["X"]
        if y_clean.nunique() != 2:
            return _blocked("TARGET_SINGLE_CLASS_AFTER_FILTERING", "After preprocessing, the target has fewer than two classes.", details=prep["details"])
        X_const = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y_clean, X_const).fit(disp=0, maxiter=100)
        coef_table = []
        for term in model.params.index:
            coef = float(model.params[term])
            coef_table.append({
                "term": str(term),
                "coef_log_odds": _round_or_none(coef),
                "odds_ratio": _round_or_none(np.exp(coef)),
                "std_err": _round_or_none(model.bse[term]),
                "z": _round_or_none(model.tvalues[term]),
                "p_value": _round_or_none(model.pvalues[term]),
            })
        return _ok("Logistic regression completed successfully.", {
            **prep["details"],
            "model_type": "Binary logistic regression",
            "pseudo_r_squared": _round_or_none(model.prsquared),
            "aic": _round_or_none(model.aic),
            "bic": _round_or_none(model.bic) if hasattr(model, "bic") else None,
            "nobs": int(model.nobs),
            "coef_table": coef_table,
        })
    except Exception as e:
        return _failed("LOGISTIC_REGRESSION_EXCEPTION", "Logistic regression failed.", e, suggested_next_actions=["Check separation, reduce predictors, or inspect binary target coding."])


# ==========================================================
# Plotting tools
# ==========================================================

@registry.register()
def generate_scatterplot(context):
    """Generate a safe scatterplot. Args: x_column, y_column."""
    try:
        df = _standardize_dataframe(_load_df(context))
        x_col = _get_arg(context, "x_column")
        y_col = _get_arg(context, "y_column")
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return _blocked("PLOT_COLUMNS_NOT_FOUND", "Valid x_column and y_column are required.", details={"x_column": x_col, "y_column": y_col})
        plot_df = df[[x_col, y_col]].copy().replace([np.inf, -np.inf], np.nan).dropna()
        if len(plot_df) < 2:
            return _blocked("TOO_FEW_POINTS_FOR_PLOT", "At least 2 complete rows are required for a scatterplot.", details={"n_complete": int(len(plot_df))})
        output_dir = getattr(context, "workspace_dir", "artifacts") or "artifacts"
        os.makedirs(output_dir, exist_ok=True)
        image_name = f"scatter_{uuid.uuid4().hex[:8]}.png"
        save_path = os.path.join(output_dir, image_name)
        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.scatterplot(data=plot_df, x=x_col, y=y_col)
        else:
            plt.scatter(plot_df[x_col], plot_df[y_col])
        plt.title(f"Scatter plot of {y_col} vs {x_col}")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        return _ok("Scatterplot generated successfully.", {"x_column": x_col, "y_column": y_col, "n_plotted": int(len(plot_df))}, artifacts=[{"type": "png", "path": save_path, "name": image_name}])
    except Exception as e:
        plt.close("all")
        return _failed("SCATTERPLOT_EXCEPTION", "Scatterplot generation failed.", e)


@registry.register()
def generate_residual_histogram(context):
    """Generate residual histogram from a fitted OLS model. Args: target_col, feature_cols."""
    try:
        from scipy import stats

        df = _load_df(context)

        prep = prepare_regression_data(
            df,
            _get_arg(context, "target_col"),
            _get_arg(context, "feature_cols", []),
            max_missing_rate=float(_get_arg(context, "max_missing_rate", 0.40)),
            max_categorical_levels=int(_get_arg(context, "max_categorical_levels", 10)),
            min_n_per_parameter=int(_get_arg(context, "min_n_per_parameter", 3)),
        )

        if prep.get("status") != "ok":
            return prep

        y, X = prep["y"], prep["X"]
        model = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

        residuals = (
            pd.Series(model.resid)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(residuals) < 3:
            return _blocked(
                "TOO_FEW_RESIDUALS_FOR_PLOT",
                "Too few valid residuals to generate a histogram.",
                details={
                    **prep["details"],
                    "n_residuals": int(len(residuals)),
                },
            )

        # =====================================================
        # Residual summary for evidence-based interpretation
        # =====================================================
        residual_mean = float(residuals.mean())
        residual_std = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
        residual_min = float(residuals.min())
        residual_max = float(residuals.max())

        if residual_std > 0:
            standardized_residuals = (residuals - residual_mean) / residual_std
            outliers_abs_2sd = int((standardized_residuals.abs() > 2).sum())
            outliers_abs_3sd = int((standardized_residuals.abs() > 3).sum())
        else:
            standardized_residuals = residuals * 0
            outliers_abs_2sd = 0
            outliers_abs_3sd = 0

        residual_skewness = (
            float(stats.skew(residuals, bias=False))
            if len(residuals) >= 3
            else None
        )

        residual_kurtosis = (
            float(stats.kurtosis(residuals, fisher=True, bias=False))
            if len(residuals) >= 4
            else None
        )

        diagnostic_flags = []

        # Skewness flags. Thresholds are intentionally conservative.
        if residual_skewness is not None:
            if residual_skewness <= -0.5:
                diagnostic_flags.append("left_skew_detected")
            elif residual_skewness >= 0.5:
                diagnostic_flags.append("right_skew_detected")

        # Outlier flags.
        if outliers_abs_3sd > 0:
            diagnostic_flags.append("possible_extreme_residual_outliers_abs_3sd")
        elif outliers_abs_2sd > 0:
            diagnostic_flags.append("possible_moderate_residual_outliers_abs_2sd")

        # Tail flag using Fisher kurtosis.
        # Fisher kurtosis > 0 means heavier than normal; >1 is a useful practical flag.
        if residual_kurtosis is not None and residual_kurtosis > 1:
            diagnostic_flags.append("heavy_tails_possible")

        if residual_std == 0:
            diagnostic_flags.append("zero_residual_variance")

        residual_summary = {
            "n_residuals": int(len(residuals)),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "residual_min": residual_min,
            "residual_max": residual_max,
            "residual_skewness": residual_skewness,
            "residual_kurtosis_fisher": residual_kurtosis,
            "outliers_abs_2sd": outliers_abs_2sd,
            "outliers_abs_3sd": outliers_abs_3sd,
            "diagnostic_flags": diagnostic_flags,
            "interpretation_guardrail": (
                "A residual histogram was generated for visual inspection. "
                "Do not claim residual normality based only on the histogram artifact. "
                "Use residual_summary, diagnostic_flags, Q-Q plot, or formal tests for stronger conclusions."
            ),
        }

        # =====================================================
        # Plot artifact
        # =====================================================
        output_dir = getattr(context, "workspace_dir", "artifacts") or "artifacts"
        os.makedirs(output_dir, exist_ok=True)

        image_name = f"residual_hist_{uuid.uuid4().hex[:8]}.png"
        save_path = os.path.join(output_dir, image_name)

        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=min(20, max(5, int(np.sqrt(len(residuals))))))
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Residual Histogram")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        details = {
            **prep["details"],
            **residual_summary,
        }

        return _ok(
            "Residual histogram generated successfully.",
            details,
            artifacts=[
                {
                    "type": "png",
                    "path": save_path,
                    "name": image_name,
                }
            ],
        )

    except Exception as e:
        plt.close("all")
        return _failed(
            "RESIDUAL_HISTOGRAM_EXCEPTION",
            "Residual histogram generation failed.",
            e,
        )

@registry.register(requires_confirmation=True)
def generate_chart(context):
    """
    Execute user/LLM-provided plotting code in a limited local namespace.

    Args:
    - code: Python plotting code. It may use df, save_path, plt, sns, pd, np, sm.
    The code must save the figure to save_path.
    """
    try:
        df = _standardize_dataframe(_load_df(context))
        code_string = _get_arg(context, "code")
        if not code_string:
            return _blocked("MISSING_PLOT_CODE", "No plotting code was provided.")
        output_dir = getattr(context, "workspace_dir", "artifacts") or "artifacts"
        os.makedirs(output_dir, exist_ok=True)
        image_name = f"chart_{uuid.uuid4().hex[:8]}.png"
        save_path = os.path.join(output_dir, image_name)
        local_env = {"df": df, "save_path": save_path, "plt": plt, "sns": sns, "pd": pd, "np": np, "sm": sm}
        # NOTE: Still risky; keep requires_confirmation=True and ideally sandbox in production.
        exec(code_string, {}, local_env)
        plt.close("all")
        if not os.path.exists(save_path):
            return _blocked("PLOT_NOT_SAVED", "Code executed, but no image file was created. Ensure plt.savefig(save_path, bbox_inches='tight') is called.", details={"save_path": save_path})
        return _ok("Chart generated successfully.", {"image_name": image_name, "save_path": save_path}, artifacts=[{"type": "png", "path": save_path, "name": image_name}])
    except Exception as e:
        plt.close("all")
        return _failed("GENERATE_CHART_EXCEPTION", "Plotting code execution failed.", e)


# ==========================================================
# Bootstrap tool
# ==========================================================

@registry.register()
def run_standard_bootstrap(context):
    """Standard i.i.d. bootstrap CI for a numeric mean. Args: target_col, iterations default 1000."""
    try:
        df = _standardize_dataframe(_load_df(context))
        target_col = _get_arg(context, "target_col")
        iterations = int(_get_arg(context, "iterations", 1000))
        if not target_col or target_col not in df.columns:
            return _blocked("TARGET_NOT_FOUND", "A valid target_col is required.", details={"target_col": target_col})
        data = pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        n = len(data)
        if n < 2:
            return _blocked("INSUFFICIENT_SAMPLE_SIZE", "At least 2 numeric observations are required for bootstrap.", details={"n": int(n)})
        rng = np.random.default_rng(int(_get_arg(context, "seed", 42)))
        idx = rng.integers(0, n, size=(iterations, n))
        means = data[idx].mean(axis=1)
        lower, upper = np.percentile(means, [2.5, 97.5])
        details = {
            "method": "Standard Independent Bootstrap",
            "iterations": iterations,
            "sample_size": int(n),
            "original_mean": _round_or_none(np.mean(data)),
            "ci_95_lower": _round_or_none(lower),
            "ci_95_upper": _round_or_none(upper),
        }
        if n < 30:
            return _warning("Bootstrap completed, but sample size is small; approximation may be unstable.", details)
        return _ok("Bootstrap completed successfully.", details)
    except Exception as e:
        return _failed("BOOTSTRAP_EXCEPTION", "Bootstrap failed.", e)


# ==========================================================
# Compatibility helpers for older non-registry action pipeline
# ==========================================================

def run_inspect_dataset(action, workspace_dir):
    df = workspace_dir
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": df.dtypes.astype(str).to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "total_inf": _count_inf(df),
    }


def run_summarize_columns(action, workspace_dir):
    cols = action.arguments.get("columns", [])
    missing_cols = [c for c in cols if c not in workspace_dir.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not in dataset: {missing_cols}")
    return workspace_dir[cols].describe(include="all").replace([np.inf, -np.inf], np.nan).to_dict()


def run_linear_regression(action, workspace_dir):
    outcome = action.arguments.get("outcome")
    predictors = action.arguments.get("predictors", [])
    prep = prepare_regression_data(workspace_dir, outcome, predictors)
    if prep.get("status") != "ok":
        return prep
    model = sm.OLS(prep["y"], sm.add_constant(prep["X"], has_constant="add")).fit()
    return {
        "formula_like": f"{outcome} ~ {' + '.join(predictors)}",
        "r_squared": _round_or_none(model.rsquared),
        "adj_r_squared": _round_or_none(model.rsquared_adj),
        "f_pvalue": _round_or_none(model.f_pvalue),
        "coefficients": {str(k): _round_or_none(v) for k, v in model.params.items()},
        "p_values": {str(k): _round_or_none(v) for k, v in model.pvalues.items()},
    }


def run_t_test(action, workspace_dir):
    group_col = action.arguments.get("group_column")
    value_col = action.arguments.get("value_column")
    if group_col not in workspace_dir.columns or value_col not in workspace_dir.columns:
        raise ValueError(f"Column {group_col} or {value_col} not found.")
    groups = workspace_dir[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"t-test requires exactly two unique groups; found: {groups}")
    group1_data = pd.to_numeric(workspace_dir[workspace_dir[group_col] == groups[0]][value_col], errors="coerce").dropna()
    group2_data = pd.to_numeric(workspace_dir[workspace_dir[group_col] == groups[1]][value_col], errors="coerce").dropna()
    t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    return {
        "group_1": str(groups[0]),
        "group_1_mean": _round_or_none(group1_data.mean()),
        "group_2": str(groups[1]),
        "group_2_mean": _round_or_none(group2_data.mean()),
        "t_statistic": _round_or_none(t_stat),
        "p_value": _round_or_none(p_val),
        "significant": bool(p_val < 0.05),
    }


def run_generate_scatterplot(action, workspace_dir):
    x_col = action.arguments.get("x_column")
    y_col = action.arguments.get("y_column")
    if x_col not in workspace_dir.columns or y_col not in workspace_dir.columns:
        raise ValueError(f"Column {x_col} or {y_col} not found.")
    plot_df = _standardize_dataframe(workspace_dir[[x_col, y_col]]).dropna()
    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"scatter_{uuid.uuid4().hex[:8]}.png")
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col)
    else:
        plt.scatter(plot_df[x_col], plot_df[y_col])
    plt.title(f"Scatter plot of {y_col} vs {x_col}")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    return {"message": "Successfully generated scatter plot.", "artifact_path": plot_path, "x_axis": x_col, "y_axis": y_col, "n_plotted": int(len(plot_df))}
