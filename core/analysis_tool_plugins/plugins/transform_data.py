"""transform_data -- structured, deterministic data transformations.

Sibling of clean_data: clean_data FIXES quality issues; transform_data RESHAPES
the data for analysis (derive columns, filter rows, bin, encode, rename). Every
action is STRUCTURED (no free-form expression / eval), produces a new immutable
data version, and -- per the P4 design -- is not pre-gated; instead the impact
(rows dropped by a filter, columns exploded by one-hot, inf/NaN introduced by a
derive) is surfaced after the fact by guardrail evaluators.
"""
from typing import Any, Dict, List, Tuple
import operator
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
from core.data_versions import create_child_data_version, make_audit_event


# --------------------------------------------------------------------------
# Result helpers (same contract as the other plugins)
# --------------------------------------------------------------------------
def _ok(message, details, data_version_update=None):
    result = {"status": "ok", "message": message, "recoverable": False,
              "details": details or {}, "artifacts": []}
    if data_version_update is not None:
        result["data_version_update"] = data_version_update
        result["details"]["data_version_update"] = data_version_update
    return result


def _blocked(error_code, message, details=None, suggested_next_actions=None):
    result = {"status": "blocked", "error_code": error_code, "message": message,
              "recoverable": True, "details": details or {}, "artifacts": []}
    if suggested_next_actions:
        result["suggested_next_actions"] = suggested_next_actions
    return result


def _failed(error_code, message, exc):
    return {"status": "failed", "error_code": error_code, "message": message,
            "recoverable": True,
            "details": {"exception_type": type(exc).__name__,
                        "exception_message": str(exc)},
            "artifacts": []}


def _get_arg(context, name, default=None):
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


def _active_version_id(context):
    return (getattr(context, "active_data_version_id", None)
            or getattr(context, "current_data_version_id", None) or "unknown")


def _make_version_id():
    return f"data_v_{uuid.uuid4().hex[:8]}"


class _Blocked(Exception):
    """Raised by an action helper for a user-recoverable validation error."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# Action helpers (structured, no eval)
# --------------------------------------------------------------------------
_ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
_CMP = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
        "==": operator.eq, "!=": operator.ne}


def _operand(df, val):
    """A column name resolves to a Series; anything else must be a number."""
    if isinstance(val, str) and val in df.columns:
        return df[val]
    try:
        return float(val)
    except (TypeError, ValueError):
        raise _Blocked("BAD_OPERAND",
                       f"Operand {val!r} is neither a column name nor a number.")


def _derive(df, ctx):
    new_column = _get_arg(ctx, "new_column")
    left = _get_arg(ctx, "left")
    op = _get_arg(ctx, "op")
    right = _get_arg(ctx, "right")
    if not new_column:
        raise _Blocked("MISSING_NEW_COLUMN", "derive requires new_column.")
    if op not in _ARITH:
        raise _Blocked("BAD_OP", f"derive op must be one of {list(_ARITH)}.")
    out = df.copy()
    out[new_column] = _ARITH[op](_operand(df, left), _operand(df, right))
    col = out[new_column]
    n_inf = int(np.isinf(col.to_numpy()).sum()) if pd.api.types.is_numeric_dtype(col) else 0
    return out, {"new_column": new_column, "op": op, "left": left, "right": right,
                 "n_inf": n_inf, "n_nan": int(col.isna().sum())}


def _condition_mask(df, cond):
    if not isinstance(cond, dict):
        raise _Blocked("BAD_CONDITION", "Each filter condition must be an object.")
    col = cond.get("column")
    op = str(cond.get("op", "")).lower().strip()
    val = cond.get("value")
    if col not in df.columns:
        raise _Blocked("COLUMN_NOT_FOUND", f"Column {col!r} is not in the data.")
    s = df[col]
    if op in _CMP:
        return _CMP[op](s, val)
    if op in ("in", "not_in"):
        if not isinstance(val, (list, tuple, set)):
            raise _Blocked("BAD_VALUE", f"filter op '{op}' needs a list value.")
        m = s.isin(list(val))
        return m if op == "in" else ~m
    if op == "between":
        if not (isinstance(val, (list, tuple)) and len(val) == 2):
            raise _Blocked("BAD_VALUE", "filter op 'between' needs [low, high].")
        return s.between(val[0], val[1])
    if op == "isnull":
        return s.isna()
    if op == "notnull":
        return s.notna()
    raise _Blocked("BAD_OP", f"Unknown filter op {op!r}.")


def _filter(df, ctx):
    conditions = _get_arg(ctx, "conditions")
    if not conditions or not isinstance(conditions, list):
        raise _Blocked("MISSING_CONDITIONS", "filter requires a non-empty conditions list.")
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        mask &= _condition_mask(df, cond)
    out = df[mask].reset_index(drop=True).copy()
    return out, {"conditions": conditions, "rows_kept": int(len(out))}


def _bin(df, ctx):
    column = _get_arg(ctx, "column")
    if column not in df.columns:
        raise _Blocked("COLUMN_NOT_FOUND", f"Column {column!r} is not in the data.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise _Blocked("NOT_NUMERIC", f"bin needs a numeric column; {column!r} is not.")
    new_column = _get_arg(ctx, "new_column") or f"{column}_bin"
    edges = _get_arg(ctx, "edges")
    labels = _get_arg(ctx, "labels")
    method = str(_get_arg(ctx, "method", "width")).lower()
    out = df.copy()
    if edges:
        out[new_column] = pd.cut(out[column], bins=list(edges), labels=labels,
                                 include_lowest=True)
        used = "edges"
    else:
        n_bins = int(_get_arg(ctx, "n_bins", 4) or 4)
        if method == "quantile":
            out[new_column] = pd.qcut(out[column], q=n_bins, labels=labels,
                                      duplicates="drop")
        else:
            out[new_column] = pd.cut(out[column], bins=n_bins, labels=labels,
                                     include_lowest=True)
            method = "width"
        used = method
    # Keep NaN as NaN; render interval bins as strings so the new data version
    # is parquet-serializable (pandas Interval dtype cannot be written to parquet).
    # Explicit string/numeric labels pass through unchanged.
    notna = out[new_column].notna()
    binned = out[new_column].astype("object").map(
        lambda v: str(v) if isinstance(v, pd.Interval) else v)
    out[new_column] = binned.where(notna, np.nan)
    return out, {"column": column, "new_column": new_column, "method": used,
                 "n_bins": int(out[new_column].dropna().nunique())}


def _encode(df, ctx):
    column = _get_arg(ctx, "column")
    if column not in df.columns:
        raise _Blocked("COLUMN_NOT_FOUND", f"Column {column!r} is not in the data.")
    method = str(_get_arg(ctx, "method", "onehot")).lower()
    out = df.copy()
    if method == "onehot":
        prefix = _get_arg(ctx, "prefix") or column
        dummies = pd.get_dummies(out[column], prefix=prefix).astype(int)
        out = pd.concat([out.drop(columns=[column]), dummies], axis=1)
        return out, {"column": column, "method": "onehot",
                     "new_columns": list(dummies.columns),
                     "n_new_columns": int(dummies.shape[1]),
                     "original_dropped": True}
    if method == "ordinal":
        order = _get_arg(ctx, "order")
        cats = list(order) if order else sorted(
            out[column].dropna().unique().tolist(), key=str)
        mapping = {c: i for i, c in enumerate(cats)}
        new_column = f"{column}_ordinal"
        out[new_column] = out[column].map(mapping)
        return out, {"column": column, "method": "ordinal", "order": cats,
                     "new_column": new_column}
    if method == "label":
        codes, uniques = pd.factorize(out[column])
        new_column = f"{column}_code"
        out[new_column] = codes
        return out, {"column": column, "method": "label", "new_column": new_column,
                     "n_labels": int(len(uniques))}
    raise _Blocked("BAD_METHOD", "encode method must be 'onehot', 'ordinal', or 'label'.")


def _rename(df, ctx):
    mapping = _get_arg(ctx, "mapping")
    if not mapping or not isinstance(mapping, dict):
        raise _Blocked("MISSING_MAPPING", "rename requires a {old: new} mapping.")
    missing = [k for k in mapping if k not in df.columns]
    if missing:
        raise _Blocked("COLUMN_NOT_FOUND", f"Columns not found: {missing}.")
    out = df.rename(columns=mapping)
    return out, {"mapping": mapping, "renamed": list(mapping.keys())}


_ACTIONS = {"derive": _derive, "filter": _filter, "bin": _bin,
            "encode": _encode, "rename": _rename}


# --------------------------------------------------------------------------
# Post-execution impact guardrails (transparency, never a gate)
# --------------------------------------------------------------------------
FILTER_WARN_FRACTION = 0.50
FILTER_INFO_FRACTION = 0.20
ENCODE_COLS_WARN = 20


def _gr(severity, title, message, evidence, recommendation=""):
    return {"finding_id": f"gr_{uuid.uuid4().hex[:8]}", "category": "data_transform_impact",
            "severity": severity, "title": title, "message": message,
            "evidence": evidence, "recommendation": recommendation}


def evaluate_transform_data_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    metrics = run.get("metrics", {}) or {}
    meta = run.get("metadata", {}) or {}
    info = meta.get("action_info", {}) or {}
    action_type = meta.get("action_type")

    if action_type == "filter":
        orig = metrics.get("original_n_rows") or 0
        removed = metrics.get("rows_removed") or 0
        if orig and removed:
            frac = removed / orig
            if frac >= FILTER_WARN_FRACTION:
                findings.append(_gr(
                    "warning", "Filter kept a small subset",
                    f"The filter dropped {removed} of {orig} rows "
                    f"({frac:.0%} removed); analysis runs on the remainder.",
                    {"rows_removed": removed, "original_n_rows": orig, "fraction": round(frac, 4)},
                    "Confirm the subset is the intended population."))
            elif frac >= FILTER_INFO_FRACTION:
                findings.append(_gr(
                    "info", "Rows filtered out",
                    f"The filter dropped {removed} of {orig} rows ({frac:.0%}).",
                    {"rows_removed": removed, "original_n_rows": orig, "fraction": round(frac, 4)}))

    if action_type == "encode" and info.get("method") == "onehot":
        n_new = info.get("n_new_columns") or 0
        if n_new >= ENCODE_COLS_WARN:
            findings.append(_gr(
                "warning", "One-hot encoding exploded the column count",
                f"One-hot encoding `{info.get('column')}` added {n_new} columns.",
                {"column": info.get("column"), "n_new_columns": n_new},
                "High-cardinality columns may be better as ordinal/target encoding."))

    if action_type == "derive":
        if info.get("n_inf"):
            findings.append(_gr(
                "warning", "Derived column has infinite values",
                f"`{info.get('new_column')}` contains {info.get('n_inf')} infinite "
                f"value(s) (likely division by zero).",
                {"new_column": info.get("new_column"), "n_inf": info.get("n_inf")},
                "Guard the denominator or filter the offending rows."))
        elif info.get("n_nan"):
            findings.append(_gr(
                "info", "Derived column has missing values",
                f"`{info.get('new_column')}` has {info.get('n_nan')} missing value(s).",
                {"new_column": info.get("new_column"), "n_nan": info.get("n_nan")}))

    return findings


# --------------------------------------------------------------------------
# Execute
# --------------------------------------------------------------------------
def execute_transform_data(context) -> Dict[str, Any]:
    """Structured transform. Each call creates a new immutable data version.

    Supported actions:
        action_type='derive'  : new_column = left <op> right
                                 op in (+, -, *, /); left/right are a column name or a number
        action_type='filter'  : conditions=[{column, op, value}, ...] combined with AND
                                 op in (>, >=, <, <=, ==, !=, in, not_in, between, isnull, notnull)
        action_type='bin'     : column [, new_column] [, method=width|quantile, n_bins | edges] [, labels]
        action_type='encode'  : column, method=onehot|ordinal|label [, order, prefix]
        action_type='rename'  : mapping={old: new}
    """
    try:
        df = context.load_df()
        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked("INVALID_DATAFRAME",
                            "context.load_df() did not return a valid pandas DataFrame.")

        action_type = str(_get_arg(context, "action_type", "")).lower().strip()
        action = _ACTIONS.get(action_type)
        if action is None:
            return _blocked(
                "UNSUPPORTED_TRANSFORM_ACTION",
                f"Unsupported action_type: {action_type}",
                details={"action_type": action_type},
                suggested_next_actions=[
                    "Use action_type in: derive, filter, bin, encode, rename."])

        original_shape = df.shape
        try:
            new_df, action_info = action(df, context)
        except _Blocked as b:
            return _blocked(b.code, str(b), details={"action_type": action_type})

        final_shape = new_df.shape
        old_version_id = _active_version_id(context)
        workspace_dir = getattr(context, "workspace_dir", None)
        description = f"Transformed data using action_type={action_type}."

        # Create a real immutable child version and switch the active version to it
        # (same protocol as materialize_sql_query_result / clean_data) so the graph
        # activates it and the next analysis tool reads the transformed data.
        version = create_child_data_version(
            df=new_df,
            workspace_dir=workspace_dir,
            parent_version_id=old_version_id,
            operation="transform_data",
            created_by="transform_data",
            description=description,
            metadata={"action_type": action_type},
        )
        new_version_id = version["version_id"]

        audit_event = make_audit_event(
            event_type="data_version_created",
            description=description,
            version_id=new_version_id,
            parent_version_id=old_version_id,
            tool_name="transform_data",
            details={
                "action_type": action_type,
                "n_rows": int(final_shape[0]),
                "n_cols": int(final_shape[1]),
            },
        )

        data_version_update = {
            # Keys the graph reads to register the new version + switch active.
            "active_data_version_id": new_version_id,
            "new_version": version,
            "audit_event": audit_event,
            # Descriptive keys kept so the report provenance (DG) reads lineage.
            "old_version_id": old_version_id,
            "new_version_id": new_version_id,
            "parent_version_id": old_version_id,
            "operation": "transform_data",
            "description": description,
            "n_rows": int(final_shape[0]),
            "n_cols": int(final_shape[1]),
            "columns": list(new_df.columns),
            "action_type": action_type,
        }

        details = {
            "action_type": action_type,
            "original_shape": str(original_shape),
            "final_shape": str(final_shape),
            "original_n_rows": int(original_shape[0]),
            "original_n_cols": int(original_shape[1]),
            "final_n_rows": int(final_shape[0]),
            "final_n_cols": int(final_shape[1]),
            "rows_removed": int(original_shape[0] - final_shape[0]),
            "cols_added": int(final_shape[1] - original_shape[1]),
            "final_columns": list(new_df.columns),
            "old_version_id": old_version_id,
            "new_version_id": new_version_id,
            "data_version_created": True,
            "action_info": action_info,
        }

        return _ok("Data transform completed and a new data version was created.",
                   details, data_version_update=data_version_update)

    except Exception as e:
        return _failed("TRANSFORM_DATA_EXCEPTION", "Data transform failed.", e)


# --------------------------------------------------------------------------
# Extractor + display
# --------------------------------------------------------------------------
def extract_transform_data(*, payload, arguments, default_title, default_summary):
    title = "Data Transform"
    metrics = compact_dict({
        "original_n_rows": payload.get("original_n_rows"),
        "final_n_rows": payload.get("final_n_rows"),
        "rows_removed": payload.get("rows_removed"),
        "original_n_cols": payload.get("original_n_cols"),
        "final_n_cols": payload.get("final_n_cols"),
        "cols_added": payload.get("cols_added"),
    })
    tables: Dict[str, Any] = {}
    metadata = compact_dict({
        "action_type": payload.get("action_type"),
        "old_version_id": payload.get("old_version_id"),
        "new_version_id": payload.get("new_version_id"),
        "data_version_update": payload.get("data_version_update"),
        "final_columns": payload.get("final_columns"),
        "action_info": payload.get("action_info"),
    })
    summary = "Transformed the active dataset and created a new data version."
    if payload.get("action_type"):
        summary += f" Action: `{payload.get('action_type')}`."
    if payload.get("new_version_id"):
        summary += f" New version: `{payload.get('new_version_id')}`."
    return title, summary, metrics, tables, metadata


TRANSFORM_DATA_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "original_n_rows": "Original rows", "final_n_rows": "Final rows",
            "rows_removed": "Rows removed", "original_n_cols": "Original columns",
            "final_n_cols": "Final columns", "cols_added": "Columns added",
        },
        formatters={k: (lambda x: format_number(x, digits=0)) for k in (
            "original_n_rows", "final_n_rows", "rows_removed",
            "original_n_cols", "final_n_cols", "cols_added")},
        order=["original_n_rows", "final_n_rows", "rows_removed",
               "original_n_cols", "final_n_cols", "cols_added"],
    ),
    tables={},
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="transform_data",
    display_name="Data Transform",
    requires_confirmation=False,   # P4: not pre-gated; guardrails report impact
    argument_schema=ArgumentSchema(
        required={"action_type": str},
        optional={
            "new_column": object, "left": object, "op": object, "right": object,
            "conditions": object, "column": object, "method": object,
            "n_bins": object, "edges": object, "labels": object,
            "order": object, "prefix": object, "mapping": object,
        },
        column_args=[],
        column_list_args=[],
        allow_all_columns=True,
    ),
    execute=execute_transform_data,
    extractor=extract_transform_data,
    guardrail_evaluators=[evaluate_transform_data_guardrails],
    display_config=TRANSFORM_DATA_DISPLAY,
))