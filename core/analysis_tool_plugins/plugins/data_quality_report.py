"""
data_quality_report -- a holistic, read-only data-quality diagnosis.

The entry point of the data-preparation workflow: it does not change the data,
it tells the agent (and the human) WHAT is wrong, so a clean/transform plan can
be proposed. Every check is deterministic Python over the active data version.

Each finding is an issue record:
    {column, issue, severity, detail, metric, suggested_action}
plus a per-column diagnostics table. Pairs with missingness_report (which is the
deep dive on missingness); this tool is the one-stop pre-analysis health check.
"""

from typing import Any, Dict, List, Tuple

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


# Thresholds (kept explicit so behaviour is auditable and tunable).
MISSING_HIGH = 0.20
MISSING_MED = 0.05
DUP_HIGH_RATE = 0.05
TEXT_AS_NUMBER_FRAC = 0.80
TEXT_AS_DATE_FRAC = 0.80
HIGH_CARD_RATIO = 0.90
OUTLIER_MIN_N = 10            # need at least this many values to judge outliers
OUTLIER_RATE_FLAG = 0.01      # or flag when this share (>=3) are 1.5*IQR outliers

# Strings that represent "missing" but are not NaN.
MISSING_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "tbd", "n.a.",
}


def _ok(message: str, details: Dict[str, Any], artifacts=None):
    return {"status": "ok", "message": message, "recoverable": False,
            "details": details or {}, "artifacts": artifacts or []}


def _blocked(error_code: str, message: str, details=None, suggested_next_actions=None):
    result = {"status": "blocked", "error_code": error_code, "message": message,
              "recoverable": True, "details": details or {}, "artifacts": []}
    if suggested_next_actions:
        result["suggested_next_actions"] = suggested_next_actions
    return result


def _failed(error_code: str, message: str, exc: Exception):
    return {"status": "failed", "error_code": error_code, "message": message,
            "recoverable": True,
            "details": {"exception_type": type(exc).__name__,
                        "exception_message": str(exc)}, "artifacts": []}


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


def _select_columns(df: pd.DataFrame, cols: Any) -> List[str]:
    if cols is None or cols == "all":
        return df.columns.tolist()
    if isinstance(cols, str):
        cols = [cols]
    if not isinstance(cols, list):
        raise ValueError("columns must be 'all', a column name, or a list of column names.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    return cols


# --------------------------------------------------------------------------
# Per-check helpers (each returns a metric; the caller decides severity)
# --------------------------------------------------------------------------
def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _frac_numeric(s: pd.Series) -> float:
    nn = s.dropna().astype(str).str.strip()
    if nn.empty:
        return 0.0
    cleaned = nn.str.replace(r"[,$%\s]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    return float(parsed.notna().mean())


def _frac_datelike(s: pd.Series) -> float:
    nn = s.dropna().astype(str).str.strip().head(300)
    if nn.empty:
        return 0.0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(nn, errors="coerce")
    return float(parsed.notna().mean())


def _norm_cat_nunique(s: pd.Series) -> int:
    return int(s.dropna().astype(str).str.strip().str.lower().nunique())


def _n_outliers(s: pd.Series) -> Tuple[int, int, int]:
    """Returns (n_outliers at 1.5*IQR, n_values, n_far_out at 3*IQR)."""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(x))
    if n < OUTLIER_MIN_N:
        return 0, n, 0
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return 0, n, 0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    far_lo, far_hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    n_out = int(((x < lo) | (x > hi)).sum())
    n_far = int(((x < far_lo) | (x > far_hi)).sum())
    return n_out, n, n_far


def _mixed_types(s: pd.Series) -> bool:
    kinds = {type(v).__name__ for v in s.dropna().head(1000)}
    return len(kinds) > 1


def _has_missing_tokens(s: pd.Series) -> int:
    vals = s.dropna().astype(str).str.strip().str.lower()
    return int(vals.isin(MISSING_TOKENS).sum())


def _infer_role(s: pd.Series, name: str, n: int) -> str:
    nun = int(s.nunique(dropna=True))
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if nun <= 1:
        return "constant"
    if str(name).lower().endswith("_id") or str(name).lower() == "id" or (n and nun == n):
        return "id_like"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if nun <= max(20, int(n * 0.5)):
        return "categorical"
    return "text"


# --------------------------------------------------------------------------
# Execute
# --------------------------------------------------------------------------
def execute_data_quality_report(context) -> Dict[str, Any]:
    """Diagnose data-quality issues across the active dataset (read-only)."""
    try:
        df = context.load_df()
        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked("INVALID_DATAFRAME",
                            "context.load_df() did not return a valid pandas DataFrame.")
        try:
            cols = _select_columns(df, _get_arg(context, "columns", "all"))
        except ValueError as e:
            return _blocked("COLUMNS_NOT_FOUND", str(e),
                            details={"available_columns": list(df.columns)},
                            suggested_next_actions=["Retry with valid column names."])

        n = int(df.shape[0])
        issues: List[Dict[str, Any]] = []
        diagnostics: List[Dict[str, Any]] = []

        def add(column, issue, severity, detail, metric, action):
            issues.append({"column": str(column), "issue": issue, "severity": severity,
                           "detail": detail, "metric": metric, "suggested_action": action})

        # whole-table: duplicate rows
        dup_rows = int(df.duplicated().sum())
        if dup_rows > 0:
            sev = "high" if (n and dup_rows / n >= DUP_HIGH_RATE) else "medium"
            add("(table)", "duplicate_rows", sev,
                f"{dup_rows} fully duplicated row(s).", dup_rows,
                "Drop exact duplicate rows (dedup).")

        for col in cols:
            s = df[col]
            role = _infer_role(s, col, n)
            n_out, n_num, n_far = _n_outliers(s)
            diagnostics.append({
                "column": str(col), "dtype": str(s.dtype),
                "missing_rate": round(float(s.isna().mean()), 6),
                "unique_count": int(s.nunique(dropna=True)),
                "role": role, "n_outliers": n_out,
            })

            # missing
            mrate = float(s.isna().mean())
            if mrate > 0:
                sev = ("high" if mrate >= MISSING_HIGH else
                       "medium" if mrate >= MISSING_MED else "low")
                add(col, "missing_values", sev,
                    f"{mrate:.1%} of values are missing.", round(mrate, 4),
                    "Impute (mean/median/mode/constant) or drop, per downstream use.")

            # constant column
            if int(s.nunique(dropna=True)) <= 1:
                add(col, "constant_column", "high",
                    "Column has a single value (no information).", 1,
                    "Drop the column.")
                continue   # other checks moot on a constant column

            if _is_text(s):
                # numeric stored as text
                fnum = _frac_numeric(s)
                if fnum >= TEXT_AS_NUMBER_FRAC:
                    add(col, "numeric_stored_as_text", "medium",
                        f"{fnum:.0%} of values parse as numbers but the column is text.",
                        round(fnum, 4), "Cast to a numeric type.")
                else:
                    # date stored as text (only if not numeric-like)
                    fdate = _frac_datelike(s)
                    if fdate >= TEXT_AS_DATE_FRAC:
                        add(col, "date_stored_as_text", "medium",
                            f"{fdate:.0%} of values parse as dates but the column is text.",
                            round(fdate, 4), "Parse to a datetime type.")

                # inconsistent categories (case/whitespace)
                nun = int(s.nunique(dropna=True))
                nun_norm = _norm_cat_nunique(s)
                if nun_norm < nun:
                    add(col, "inconsistent_categories", "medium",
                        f"{nun - nun_norm} categor(y/ies) differ only by case/whitespace "
                        f"({nun} -> {nun_norm} after normalising).", nun - nun_norm,
                        "Standardise category labels (trim/casing).")

                # disguised missing tokens
                n_tok = _has_missing_tokens(s)
                if n_tok > 0:
                    add(col, "disguised_missing", "low",
                        f"{n_tok} value(s) like 'NA'/'?'/'unknown' represent missing.",
                        n_tok, "Convert missing tokens to NaN, then handle as missing.")

                # mixed python types
                if _mixed_types(s):
                    add(col, "mixed_types", "medium",
                        "Column mixes value types (e.g. numbers and strings).", None,
                        "Normalise to a single type.")

                # high cardinality (likely identifier / free text)
                if n and (s.nunique(dropna=True) / n) >= HIGH_CARD_RATIO and role != "id_like":
                    add(col, "high_cardinality", "low",
                        f"{s.nunique(dropna=True)} distinct values "
                        f"({s.nunique(dropna=True) / n:.0%} of rows) -- likely an identifier "
                        "or free text, not a grouping variable.", int(s.nunique(dropna=True)),
                        "Treat as an identifier; exclude from grouping/encoding.")

            # outliers (numeric): flag a single far-out value (>3*IQR) OR many mild ones
            if n_num >= OUTLIER_MIN_N and (n_far >= 1 or
                                           n_out >= max(3, int(OUTLIER_RATE_FLAG * n_num))):
                rate = n_out / n_num
                extra = f", {n_far} far beyond 3*IQR" if n_far else ""
                add(col, "outliers", "low",
                    f"{n_out} value(s) ({rate:.1%}) lie beyond 1.5*IQR{extra}.", n_out,
                    "Review; consider winsorising/clipping or flagging, not silent removal.")

        sev_counts = {s: sum(1 for i in issues if i["severity"] == s)
                      for s in ("high", "medium", "low")}
        issues.sort(key=lambda i: {"high": 0, "medium": 1, "low": 2}[i["severity"]])

        return _ok(
            f"Data-quality report: {len(issues)} issue(s) across {df.shape[1]} columns.",
            {
                "shape": {"rows": n, "columns": int(df.shape[1])},
                "n_issues": len(issues),
                "severity_counts": sev_counts,
                "duplicate_rows": dup_rows,
                "issues": issues,
                "column_diagnostics": diagnostics,
            },
        )
    except Exception as e:
        return _failed("DATA_QUALITY_REPORT_EXCEPTION", "Data-quality report failed.", e)


class _DfContext:
    """Minimal context so the diagnosis can run on a bare DataFrame (no graph)."""
    def __init__(self, df, columns="all"):
        self._df = df
        self._columns = columns

    def load_df(self):
        return self._df

    def get_arg(self, name, default=None):
        return self._columns if name == "columns" else default


def diagnose(df, columns="all"):
    """Context-free diagnosis: run the data-quality checks on a DataFrame and return
    the details payload. Lets an orchestrator force a quality check before analysis
    without going through the agent graph."""
    result = execute_data_quality_report(_DfContext(df, columns))
    return result.get("details", {}) if result.get("status") == "ok" else result


# --------------------------------------------------------------------------
# Extractor (payload -> title/summary/metrics/tables/metadata)
# --------------------------------------------------------------------------
def extract_data_quality_report(*, payload, arguments, default_title, default_summary):
    shape = payload.get("shape", {}) or {}
    sev = payload.get("severity_counts", {}) or {}
    metrics = compact_dict({
        "n_rows": shape.get("rows"),
        "n_columns": shape.get("columns"),
        "n_issues": payload.get("n_issues"),
        "n_high": sev.get("high"),
        "n_medium": sev.get("medium"),
        "n_low": sev.get("low"),
        "duplicate_rows": payload.get("duplicate_rows"),
    })
    tables: Dict[str, Any] = {}
    if payload.get("issues"):
        tables["issues"] = payload["issues"]
    if payload.get("column_diagnostics"):
        tables["column_diagnostics"] = payload["column_diagnostics"]
    summary = (f"Found {payload.get('n_issues', 0)} data-quality issue(s) "
               f"({sev.get('high', 0)} high, {sev.get('medium', 0)} medium, "
               f"{sev.get('low', 0)} low).")
    return "Data Quality Report", summary, metrics, tables, {}


DATA_QUALITY_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={"n_rows": "Rows", "n_columns": "Columns", "n_issues": "Issues",
                "n_high": "High severity", "n_medium": "Medium severity",
                "n_low": "Low severity", "duplicate_rows": "Duplicate rows"},
        order=["n_rows", "n_columns", "n_issues", "n_high", "n_medium", "n_low",
               "duplicate_rows"],
    ),
    tables={
        "issues": TableDisplayConfig(
            column_labels={"column": "Column", "issue": "Issue", "severity": "Severity",
                           "detail": "Detail", "metric": "Metric",
                           "suggested_action": "Suggested action"},
            column_order=["severity", "column", "issue", "detail", "suggested_action"],
        ),
        "column_diagnostics": TableDisplayConfig(
            column_labels={"column": "Column", "dtype": "Data type",
                           "missing_rate": "Missing rate", "unique_count": "Unique values",
                           "role": "Inferred role", "n_outliers": "Outliers"},
            column_formatters={"missing_rate": lambda x: format_number(x, digits=4)},
            column_order=["column", "dtype", "role", "missing_rate", "unique_count",
                          "n_outliers"],
        ),
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="data_quality_report",
    display_name="Data Quality Report",
    evidence_categories=["data_quality"],
    evidence_category_roles={"data_quality": "pre_analysis_check"},
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={"columns": object},
        column_args=[],
        column_list_args=["columns"],
        allow_all_columns=True,
    ),
    execute=execute_data_quality_report,
    extractor=extract_data_quality_report,
    guardrail_evaluators=[],
    display_config=DATA_QUALITY_DISPLAY,
))