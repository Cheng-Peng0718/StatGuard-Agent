from typing import Any, Dict, List, Tuple
import uuid

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin


def _ok(message: str, details: Dict[str, Any], artifacts=None, data_version_update=None):
    result = {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }

    # Put it both places during migration.
    # execution.py preserves top-level data_version_update into payload,
    # and details also carries it for downstream compatibility.
    if data_version_update is not None:
        result["data_version_update"] = data_version_update
        result["details"]["data_version_update"] = data_version_update

    return result


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


def _active_version_id(context) -> str:
    return (
        getattr(context, "active_data_version_id", None)
        or getattr(context, "current_data_version_id", None)
        or "unknown"
    )


def _make_version_id() -> str:
    return f"data_v_{uuid.uuid4().hex[:8]}"


def _normalize_columns_arg(columns):
    if columns is None:
        return []

    if isinstance(columns, str):
        return [columns]

    if isinstance(columns, list):
        return columns

    return []


def _selected_columns(df: pd.DataFrame, columns_arg):
    columns = _normalize_columns_arg(columns_arg)

    if not columns:
        return df.columns.tolist()

    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"Columns not found: {missing}")

    return columns


def _count_inf(df: pd.DataFrame) -> int:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return 0

    try:
        return int(np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum())
    except Exception:
        return 0


def _missing_by_columns(df: pd.DataFrame, columns: list[str]) -> Dict[str, int]:
    return {
        str(col): int(df[col].isna().sum())
        for col in columns
        if col in df.columns
    }


def _inf_by_columns(df: pd.DataFrame, columns: list[str]) -> Dict[str, int]:
    out = {}

    for col in columns:
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            out[str(col)] = 0
            continue

        try:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            out[str(col)] = int(np.isinf(arr).sum())
        except Exception:
            out[str(col)] = 0

    return out


def _impute_numeric_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].mean(skipna=True)
            df[col] = df[col].fillna(value)

    return df


def _impute_numeric_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median(skipna=True)
            df[col] = df[col].fillna(value)

    return df


def _impute_mode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing with the most frequent value (works for any dtype)."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        modes = df[col].mode(dropna=True)
        if len(modes):
            df[col] = df[col].fillna(modes.iloc[0])
    return df


def _impute_constant(df: pd.DataFrame, columns: list[str], fill_value: Any) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df


def _cast(df: pd.DataFrame, columns: list[str], strategy: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cast columns to a type. Coercion failures become NaN and are counted
    (a high failure count is a signal the cast was wrong)."""
    df = df.copy()
    info: Dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            continue
        na_before = int(df[col].isna().sum())
        if strategy == "numeric":
            cleaned = df[col].astype(str).str.replace(r"[,$%\s]", "", regex=True)
            df[col] = pd.to_numeric(cleaned, errors="coerce")
        elif strategy in ("datetime", "date"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce")
        elif strategy in ("category", "categorical"):
            df[col] = df[col].astype("category")
        elif strategy in ("string", "str"):
            df[col] = df[col].astype("string")
        else:
            raise ValueError(
                "For action_type='cast', strategy must be "
                "'numeric', 'datetime', 'category', or 'string'.")
        na_after = int(df[col].isna().sum())
        info[str(col)] = {"new_dtype": str(df[col].dtype),
                          "coercion_failures": max(0, na_after - na_before)}
    return df, info


def _dedup(df: pd.DataFrame, subset) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    before = len(df)
    new_df = df.drop_duplicates(subset=subset or None, keep="first").reset_index(drop=True)
    return new_df, {"rows_removed": int(before - len(new_df)),
                    "subset": list(subset) if subset else None}


def _standardize_categories(df: pd.DataFrame, columns: list[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Collapse category labels that differ only by case/whitespace, mapping each
    case-folded group to its most frequent surface form (keeps a readable label)."""
    df = df.copy()
    info: Dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            continue
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue
        s = df[col]
        stripped = s.map(lambda x: " ".join(str(x).split()) if isinstance(x, str) else x)
        notna = stripped.notna()
        key = stripped.where(~notna, stripped[notna].astype(str).str.casefold())
        tmp = pd.DataFrame({"k": key[notna], "v": stripped[notna]})
        surface = {k: g["v"].mode().iloc[0] for k, g in tmp.groupby("k")}
        n_before = int(s.nunique(dropna=True))
        df[col] = stripped.where(~notna, key.map(surface))
        n_after = int(df[col].nunique(dropna=True))
        info[str(col)] = {"labels_before": n_before, "labels_after": n_after}
    return df, info


def _clip_outliers(df: pd.DataFrame, columns: list[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clip numeric values to the 1.5*IQR fence (winsorise outliers, not remove rows)."""
    df = df.copy()
    info: Dict[str, Any] = {}
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = df[col]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_clipped = int(((x < lo) | (x > hi)).sum())
        df[col] = x.clip(lower=lo, upper=hi)
        info[str(col)] = {"n_clipped": n_clipped,
                          "lower": float(lo), "upper": float(hi)}
    return df, info


def _impute_ffill_bfill(df: pd.DataFrame, columns: list[str], how: str) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].ffill() if how == "ffill" else df[col].bfill()
    return df


# --------------------------------------------------------------------------
# Per-action review gate (P3) + post-execution impact guardrails
# --------------------------------------------------------------------------
DROP_ROWS_REVIEW_FRACTION = 0.05    # drop that would discard >5% of rows -> review
IMPUTE_REVIEW_FRACTION = 0.20       # impute a column >20% missing -> review
ROW_LOSS_WARN_FRACTION = 0.20       # actually removed >20% of rows -> warning
ROW_LOSS_INFO_FRACTION = 0.05       # actually removed >5% -> info


def _profile_columns(profile) -> Tuple[Dict[str, Any], int]:
    """Pull (columns_info, n_rows) from a DatasetProfile (object or dict)."""
    if profile is None:
        return {}, 0
    cols = getattr(profile, "columns", None)
    if cols is None and isinstance(profile, dict):
        cols = profile.get("columns")
    n_rows = getattr(profile, "n_rows", None)
    if n_rows is None and isinstance(profile, dict):
        n_rows = profile.get("n_rows")
    return (cols if isinstance(cols, dict) else {}), int(n_rows or 0)


def _missing_rate(cols_info: Dict[str, Any], col: str) -> float:
    info = cols_info.get(col)
    if isinstance(info, dict):
        return float(info.get("missing_rate", 0.0) or 0.0)
    return float(getattr(info, "missing_rate", 0.0) or 0.0)


def clean_data_confirmation_policy(action, profile, state):
    """Only DESTRUCTIVE clean actions need human review; the rest run freely.

    Pre-execution, so row-loss is *estimated* from the profile's per-column
    missing rates (the post-exec guardrail reports the exact impact).
    """
    args = getattr(action, "arguments", {}) or {}
    action_type = str(args.get("action_type", "")).lower().strip()
    selected = _normalize_columns_arg(args.get("columns"))

    cols_info, _ = _profile_columns(profile)
    targets = selected or list(cols_info.keys())

    if action_type == "drop":
        worst = max((_missing_rate(cols_info, c) for c in targets), default=0.0)
        if worst > DROP_ROWS_REVIEW_FRACTION:
            return True, (
                f"Dropping rows would discard at least ~{worst * 100:.0f}% of the data "
                f"(missing values in the targeted column(s)). Confirm before proceeding."
            )
        return False, ""

    if action_type == "impute":
        worst_col, worst = "", 0.0
        for c in targets:
            r = _missing_rate(cols_info, c)
            if r > worst:
                worst_col, worst = c, r
        if worst > IMPUTE_REVIEW_FRACTION:
            return True, (
                f"Imputing `{worst_col}` would fill ~{worst * 100:.0f}% of its values "
                f"(over 20% missing); this can distort the column. Confirm before proceeding."
            )
        return False, ""

    # cast / dedup / standardize / clip -> lower risk; allowed (guardrails report impact).
    return False, ""


def _gr_finding(severity, title, message, evidence, recommendation=""):
    return {
        "finding_id": f"gr_{uuid.uuid4().hex[:8]}",
        "category": "data_cleaning_impact",
        "severity": severity,
        "title": title,
        "message": message,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def evaluate_clean_data_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Post-execution transparency: report the ACTUAL impact of a clean step."""
    findings: List[Dict[str, Any]] = []
    metrics = run.get("metrics", {}) or {}
    meta = run.get("metadata", {}) or {}
    action_info = meta.get("action_info", {}) or {}

    orig = metrics.get("original_n_rows") or 0
    removed = metrics.get("rows_removed") or 0
    if orig and removed:
        frac = removed / orig
        if frac >= ROW_LOSS_WARN_FRACTION:
            findings.append(_gr_finding(
                "warning", "Large row reduction",
                f"This step removed {removed} of {orig} rows ({frac:.0%}).",
                {"rows_removed": removed, "original_n_rows": orig, "fraction": round(frac, 4)},
                "Confirm this loss is intended; consider imputation or a narrower filter."))
        elif frac >= ROW_LOSS_INFO_FRACTION:
            findings.append(_gr_finding(
                "info", "Rows removed",
                f"This step removed {removed} of {orig} rows ({frac:.0%}).",
                {"rows_removed": removed, "original_n_rows": orig, "fraction": round(frac, 4)}))

    for col, ci in (action_info.get("cast", {}) or {}).items():
        cf = (ci or {}).get("coercion_failures", 0) or 0
        if cf > 0:
            findings.append(_gr_finding(
                "warning", "Values lost in type cast",
                f"Casting `{col}` left {cf} value(s) unparseable; they became missing.",
                {"column": col, "coercion_failures": cf},
                "Inspect the unparseable values; the chosen type may be wrong for this column."))

    for col, ci in (action_info.get("clip", {}) or {}).items():
        nc = (ci or {}).get("n_clipped", 0) or 0
        if nc > 0:
            findings.append(_gr_finding(
                "info", "Outliers clipped",
                f"Clipped {nc} outlier value(s) in `{col}` to the 1.5*IQR fence.",
                {"column": col, "n_clipped": nc}))

    return findings


def execute_clean_data(context) -> Dict[str, Any]:
    """
    Mutating data-cleaning tool. Each call creates a new immutable data version.

    Supported actions:
        action_type='drop',        strategy='rows'
        action_type='impute',      strategy='mean' | 'median' | 'mode'
                                    | 'ffill' | 'bfill' | 'constant'
                                    (constant also needs fill_value)
        action_type='cast',        strategy='numeric' | 'datetime' | 'category' | 'string'
        action_type='dedup',       strategy='rows'   (dedups on `columns` if given, else full row)
        action_type='standardize', strategy='categories'
        action_type='clip',        strategy='outliers'  (winsorise to 1.5*IQR fence)

    Args:
        action_type, strategy: see above.
        columns: optional list of columns. If omitted, all columns are considered.
        fill_value: required for impute/constant.
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        action_type = str(_get_arg(context, "action_type", "")).lower().strip()
        strategy = str(_get_arg(context, "strategy", "")).lower().strip()
        columns_arg = _get_arg(context, "columns", None)

        if not action_type:
            return _blocked(
                "MISSING_CLEAN_ACTION",
                "action_type is required.",
                suggested_next_actions=[
                    "Use action_type='drop' or action_type='impute'."
                ],
            )

        try:
            cols = _selected_columns(df, columns_arg)
        except ValueError as e:
            return _blocked(
                "COLUMNS_NOT_FOUND",
                str(e),
                details={
                    "requested_columns": columns_arg,
                    "available_columns": list(df.columns),
                },
                suggested_next_actions=[
                    "Inspect dataset columns and retry with valid column names."
                ],
            )

        original_shape = tuple(df.shape)
        total_missing_before = int(df.isna().sum().sum())
        total_inf_before = _count_inf(df)
        selected_missing_before = _missing_by_columns(df, cols)
        selected_inf_before = _inf_by_columns(df, cols)

        # Treat inf as missing before cleaning.
        work = df.copy().replace([np.inf, -np.inf], np.nan)
        action_info: Dict[str, Any] = {}

        if action_type == "drop":
            if strategy not in {"rows", "row", "drop_rows"}:
                return _blocked(
                    "UNSUPPORTED_CLEAN_STRATEGY",
                    "For action_type='drop', only strategy='rows' is supported.",
                    details={"action_type": action_type, "strategy": strategy},
                )
            new_df = work.dropna(subset=cols).copy()
            normalized_strategy = "rows"

        elif action_type == "impute":
            if strategy == "mean":
                new_df = _impute_numeric_mean(work, cols)
            elif strategy == "median":
                new_df = _impute_numeric_median(work, cols)
            elif strategy == "mode":
                new_df = _impute_mode(work, cols)
            elif strategy in ("ffill", "bfill"):
                new_df = _impute_ffill_bfill(work, cols, strategy)
            elif strategy == "constant":
                fill_value = _get_arg(context, "fill_value", None)
                if fill_value is None:
                    return _blocked(
                        "MISSING_FILL_VALUE",
                        "action_type='impute', strategy='constant' requires fill_value.",
                        details={"action_type": action_type, "strategy": strategy},
                    )
                new_df = _impute_constant(work, cols, fill_value)
                action_info["fill_value"] = fill_value
            else:
                return _blocked(
                    "UNSUPPORTED_CLEAN_STRATEGY",
                    "For action_type='impute', strategy must be "
                    "'mean', 'median', 'mode', 'ffill', 'bfill', or 'constant'.",
                    details={"action_type": action_type, "strategy": strategy},
                )
            normalized_strategy = strategy

        elif action_type == "clip":
            if strategy not in {"outliers", "iqr", ""}:
                return _blocked(
                    "UNSUPPORTED_CLEAN_STRATEGY",
                    "For action_type='clip', only strategy='outliers' (1.5*IQR) is supported.",
                    details={"action_type": action_type, "strategy": strategy})
            new_df, clip_info = _clip_outliers(work, cols)
            action_info["clip"] = clip_info
            normalized_strategy = "outliers"

        elif action_type == "cast":
            try:
                new_df, cast_info = _cast(work, cols, strategy)
            except ValueError as e:
                return _blocked("UNSUPPORTED_CLEAN_STRATEGY", str(e),
                                details={"action_type": action_type, "strategy": strategy})
            action_info["cast"] = cast_info
            normalized_strategy = strategy

        elif action_type == "dedup":
            dedup_subset = cols if _normalize_columns_arg(columns_arg) else None
            new_df, dd_info = _dedup(work, dedup_subset)
            action_info.update(dd_info)
            normalized_strategy = "rows"

        elif action_type == "standardize":
            if strategy not in {"categories", "category", "categorical", ""}:
                return _blocked(
                    "UNSUPPORTED_CLEAN_STRATEGY",
                    "For action_type='standardize', only strategy='categories' is supported.",
                    details={"action_type": action_type, "strategy": strategy})
            new_df, std_info = _standardize_categories(work, cols)
            action_info["standardize"] = std_info
            normalized_strategy = "categories"

        else:
            return _blocked(
                "UNSUPPORTED_CLEAN_ACTION",
                f"Unsupported action_type: {action_type}",
                details={"action_type": action_type},
                suggested_next_actions=[
                    "Use action_type in: drop, impute, cast, dedup, standardize, clip."
                ],
            )

        new_df = new_df.reset_index(drop=True)

        final_shape = tuple(new_df.shape)
        total_missing_after = int(new_df.isna().sum().sum())
        total_inf_after = _count_inf(new_df)
        selected_missing_after = _missing_by_columns(new_df, cols)
        selected_inf_after = _inf_by_columns(new_df, cols)

        old_version_id = _active_version_id(context)
        new_version_id = _make_version_id()

        # This is the actual mutation.
        context.save_df(new_df)

        data_version_update = {
            "old_version_id": old_version_id,
            "new_version_id": new_version_id,
            "parent_version_id": old_version_id,
            "operation": "clean_data",
            "description": (
                f"Cleaned data using action_type={action_type}, "
                f"strategy={normalized_strategy}, columns={cols}."
            ),
            "n_rows": int(new_df.shape[0]),
            "n_cols": int(new_df.shape[1]),
            "columns": list(new_df.columns),
            "action_type": action_type,
            "strategy": normalized_strategy,
            "selected_columns": cols,
        }

        details = {
            "action_type": action_type,
            "strategy": normalized_strategy,
            "selected_columns": cols,
            "original_shape": str(original_shape),
            "final_shape": str(final_shape),
            "original_n_rows": int(original_shape[0]),
            "original_n_cols": int(original_shape[1]),
            "final_n_rows": int(final_shape[0]),
            "final_n_cols": int(final_shape[1]),
            "rows_removed": int(original_shape[0] - final_shape[0]),
            "total_missing_before": total_missing_before,
            "total_missing_after": total_missing_after,
            "total_inf_before": total_inf_before,
            "total_inf_after": total_inf_after,
            "selected_missing_before": selected_missing_before,
            "selected_missing_after": selected_missing_after,
            "selected_inf_before": selected_inf_before,
            "selected_inf_after": selected_inf_after,
            "final_columns": list(new_df.columns),
            "old_version_id": old_version_id,
            "new_version_id": new_version_id,
            "data_version_created": True,
            "action_info": action_info,
        }

        return _ok(
            "Data cleaning completed and a new data version was created.",
            details,
            data_version_update=data_version_update,
        )

    except Exception as e:
        return _failed(
            "CLEAN_DATA_EXCEPTION",
            "Data cleaning failed.",
            e,
        )


def extract_clean_data(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Data Cleaning"

    metrics = compact_dict({
        "original_n_rows": payload.get("original_n_rows"),
        "final_n_rows": payload.get("final_n_rows"),
        "rows_removed": payload.get("rows_removed"),
        "total_missing_before": payload.get("total_missing_before"),
        "total_missing_after": payload.get("total_missing_after"),
        "total_inf_before": payload.get("total_inf_before"),
        "total_inf_after": payload.get("total_inf_after"),
    })

    tables: Dict[str, Any] = {}

    selected_missing_after = payload.get("selected_missing_after", {})
    if isinstance(selected_missing_after, dict) and selected_missing_after:
        tables["selected_missing_after"] = [
            {
                "column": col,
                "missing_after": value,
            }
            for col, value in selected_missing_after.items()
        ]

    metadata = compact_dict({
        "action_type": payload.get("action_type"),
        "strategy": payload.get("strategy"),
        "selected_columns": payload.get("selected_columns"),
        "old_version_id": payload.get("old_version_id"),
        "new_version_id": payload.get("new_version_id"),
        "data_version_update": payload.get("data_version_update"),
        "final_columns": payload.get("final_columns"),
        "action_info": payload.get("action_info"),
    })

    summary = "Cleaned the active dataset and created a new data version."

    if payload.get("action_type") and payload.get("strategy"):
        summary += f" Action: `{payload.get('action_type')}` using strategy `{payload.get('strategy')}`."

    if payload.get("new_version_id"):
        summary += f" New version: `{payload.get('new_version_id')}`."

    return title, summary, metrics, tables, metadata


CLEAN_DATA_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "original_n_rows": "Original rows",
            "final_n_rows": "Final rows",
            "rows_removed": "Rows removed",
            "total_missing_before": "Missing values before",
            "total_missing_after": "Missing values after",
            "total_inf_before": "Infinite values before",
            "total_inf_after": "Infinite values after",
        },
        formatters={
            "original_n_rows": lambda x: format_number(x, digits=0),
            "final_n_rows": lambda x: format_number(x, digits=0),
            "rows_removed": lambda x: format_number(x, digits=0),
            "total_missing_before": lambda x: format_number(x, digits=0),
            "total_missing_after": lambda x: format_number(x, digits=0),
            "total_inf_before": lambda x: format_number(x, digits=0),
            "total_inf_after": lambda x: format_number(x, digits=0),
        },
        order=[
            "original_n_rows",
            "final_n_rows",
            "rows_removed",
            "total_missing_before",
            "total_missing_after",
            "total_inf_before",
            "total_inf_after",
        ],
    ),
    tables={
        "selected_missing_after": TableDisplayConfig(
            column_labels={
                "column": "Column",
                "missing_after": "Missing after cleaning",
            },
            column_order=[
                "column",
                "missing_after",
            ],
        )
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="clean_data",
    display_name="Data Cleaning",
    requires_confirmation=False,   # per-action gating via confirmation_policy (P3)
    argument_schema=ArgumentSchema(
        required={
            "action_type": str,
            "strategy": str,
        },
        optional={
            "columns": object,
            "fill_value": object,
        },
        column_args=[],
        column_list_args=[
            "columns",
        ],
        allow_all_columns=True,
    ),
    execute=execute_clean_data,
    extractor=extract_clean_data,
    guardrail_evaluators=[evaluate_clean_data_guardrails],
    confirmation_policy=clean_data_confirmation_policy,
    display_config=CLEAN_DATA_DISPLAY,
))