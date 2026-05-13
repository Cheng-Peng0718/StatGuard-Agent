import os
import uuid
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

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
            excluded_features.append({
                "column": col,
                "reason": "categorical_encoding_produced_no_columns",
                "n_levels": n_levels,
            })
            continue

        encoded_columns = dummies.columns.tolist()
        prefix = f"{col}_"

        encoded_levels = []
        for encoded_col in encoded_columns:
            encoded_col_str = str(encoded_col)
            if encoded_col_str.startswith(prefix):
                encoded_levels.append(encoded_col_str[len(prefix):])
            else:
                encoded_levels.append(encoded_col_str)

        observed_levels = sorted([str(x) for x in non_missing.unique()])
        reference_candidates = [
            level for level in observed_levels
            if level not in encoded_levels
        ]
        reference_level = reference_candidates[0] if reference_candidates else None

        X_parts.append(dummies)
        used_features.append({
            "column": col,
            "type": "categorical_encoded",
            "n_levels": n_levels,
            "levels": observed_levels,
            "reference_level": reference_level,
            "encoded_columns": encoded_columns,
            "encoded_level_map": {
                encoded_col: encoded_level
                for encoded_col, encoded_level in zip(encoded_columns, encoded_levels)
            },
        })

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

