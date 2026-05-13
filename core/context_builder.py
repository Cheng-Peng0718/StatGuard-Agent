import os

import pandas as pd
from core.schema import ContextPackage, DatasetProfile, ColumnProfile


def get_observation_data_version(obs):
    if not isinstance(obs, dict):
        return None

    if obs.get("data_version_id"):
        return obs.get("data_version_id")

    structured = obs.get("structured_data", {})
    if isinstance(structured, dict):
        return structured.get("data_version_id")

    return None

####### HELPER ########

def _sql_observation_extra(obs: dict) -> str:
    """
    Return compact SQL-specific context for the Supervisor.

    SQL observations are not tied to the active DataFrame data_version_id,
    so they must be exposed even when version_status is UNKNOWN_VERSION.
    Keep this compact: do not dump full previews or large payloads.
    """
    tool_name = obs.get("tool_name")
    structured = obs.get("structured_data", {}) or {}

    payload = {}
    if isinstance(structured, dict):
        payload = structured.get("payload", {}) or {}

    if not isinstance(payload, dict):
        return ""

    if tool_name == "inspect_sql_schema":
        compact_schema = payload.get("compact_schema")

        if compact_schema:
            return f"\n  SQL schema: {compact_schema}"

        tables = payload.get("tables", [])
        if isinstance(tables, list):
            parts = []

            for table in tables:
                if not isinstance(table, dict):
                    continue

                table_name = table.get("table_name")
                columns = table.get("columns", [])

                column_names = [
                    col.get("column_name")
                    for col in columns
                    if isinstance(col, dict) and col.get("column_name")
                ]

                if table_name and column_names:
                    parts.append(f"{table_name}({', '.join(column_names)})")

            if parts:
                return f"\n  SQL schema: {'; '.join(parts)}"

    if tool_name in {"run_sql_query", "materialize_sql_query_result"}:
        extras = []

        query = payload.get("query")
        if query:
            extras.append(f"SQL query: {query}")

        columns = payload.get("columns")
        if columns:
            extras.append(f"SQL result columns: {columns}")

        n_rows = payload.get("n_rows") or payload.get("n_rows_returned")
        n_cols = payload.get("n_cols") or payload.get("n_cols_returned")
        if n_rows is not None or n_cols is not None:
            extras.append(f"SQL result shape: rows={n_rows}, cols={n_cols}")

        new_data_version_id = (
            payload.get("new_data_version_id")
            or payload.get("data_version_update", {}).get("active_data_version_id")
            if isinstance(payload.get("data_version_update"), dict)
            else None
        )
        if new_data_version_id:
            extras.append(f"materialized_data_version_id: {new_data_version_id}")

        error_message = payload.get("error_message")
        if error_message:
            extras.append(f"SQL error: {error_message}")

        if extras:
            return "\n  " + "\n  ".join(extras)

    return ""

def format_observation_history(observations, active_data_version_id=None, max_items=10):
    lines = []

    for obs in (observations or [])[-max_items:]:
        if not isinstance(obs, dict):
            continue

        tool_name = obs.get("tool_name", "unknown_tool")
        status = obs.get("status", "unknown")
        success = obs.get("success")
        message = obs.get("message", "")
        summary = obs.get("summary", "")
        obs_version = get_observation_data_version(obs)

        if active_data_version_id and obs_version:
            if obs_version == active_data_version_id:
                version_status = "CURRENT"
            else:
                version_status = "STALE"
        else:
            version_status = "UNKNOWN_VERSION"

        line = (
            f"- tool={tool_name}, status={status}, success={success}, "
            f"data_version_id={obs_version}, version_status={version_status}, "
            f"message={message}, summary={summary}"
        )

        # SQL-specific compact context should be visible even when there is
        # no active DataFrame data version.
        sql_extra = _sql_observation_extra(obs)
        if sql_extra:
            line += sql_extra

        lines.append(line)

        # Only expose general payload for CURRENT DataFrame observations.
        # Do not expose full SQL payload here; SQL tools are summarized above.
        if version_status == "CURRENT" and tool_name not in {
            "inspect_sql_schema",
            "run_sql_query",
            "materialize_sql_query_result",
        }:
            structured = obs.get("structured_data", {})
            payload = structured.get("payload") if isinstance(structured, dict) else None

            if payload:
                lines.append(f"  payload={payload}")

        elif version_status == "STALE":
            lines.append(
                "  NOTE: This observation was computed on an older data version. "
                "Do not use it for current numeric answers."
            )

    return "\n".join(lines) if lines else "No previous observations."

def generate_profile(file_path: str) -> DatasetProfile:
    """Build a dataset profile report (multiple formats supported)."""

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path, engine='pyarrow')
        else:
            raise ValueError(f"Unsupported profile format: {ext}")

        def _infer_semantic_type(s: pd.Series) -> str:
            if pd.api.types.is_bool_dtype(s):
                return "boolean"
            if pd.api.types.is_numeric_dtype(s):
                return "numeric"
            if pd.api.types.is_datetime64_any_dtype(s):
                return "datetime"
            if isinstance(s.dtype, pd.CategoricalDtype):
                return "categorical"
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                non_missing = s.dropna()
                if len(non_missing) == 0:
                    return "unknown"
                numeric_rate = pd.to_numeric(non_missing, errors="coerce").notna().mean()
                if numeric_rate >= 0.85:
                    return "numeric_like"
                if non_missing.nunique() <= max(20, int(0.2 * len(s))):
                    return "categorical"
                return "text"
            return "unknown"

        def _is_id_like(col_name: str, s: pd.Series) -> bool:
            name = str(col_name).lower()
            n = len(s)
            if n == 0:
                return False
            unique_rate = s.nunique(dropna=True) / max(n, 1)
            return (
                    name in {"id", "uid", "uuid", "student_id", "record_id"}
                    or name.endswith("_id")
                    or (unique_rate > 0.95 and s.nunique(dropna=True) > 20)
            )

        columns_info = {}
        n_rows = len(df)

        for col, dtype in df.dtypes.items():
            s = df[col]
            n_missing = int(s.isnull().sum())
            n_unique = int(s.nunique(dropna=True))
            semantic_type = _infer_semantic_type(s)

            columns_info[str(col)] = {
                "name": str(col),
                "dtype": str(dtype),
                "n_missing": n_missing,
                "missing_rate": float(n_missing / max(n_rows, 1)),
                "n_unique": n_unique,
                "semantic_type": semantic_type,
                "is_numeric_like": semantic_type in {"numeric", "numeric_like"},
                "is_id_like": _is_id_like(str(col), s),
            }

        profile_dict = {
            "dataset_name": os.path.basename(file_path),
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "columns": columns_info,
        }

        return DatasetProfile(**profile_dict)

    except Exception as e:
        print(f"Profile generation failed: {str(e)}")
        raise e


def build_context(step,
                  max_steps,
                  user_request,
                  profile,
                  observations,
                  workspace_dir="./",
                  deliverable_check=None,
                  data_versions=None,
                  active_data_version_id=None,
                  data_audit_log=None,
                  analysis_coverage_brief=None,
                  ):
    """
    Build the full context text sent to the LLM.
    """

    if isinstance(profile, dict):
        rows = profile.get("n_rows", "unknown")
        cols_dict = profile.get("columns", {})
        cols = list(cols_dict.keys()) if isinstance(cols_dict, dict) else ["unable to parse column names"]
    else:
        rows = getattr(profile, "n_rows", "unknown")
        cols = list(getattr(profile, "columns", {}).keys())

    history_log = format_observation_history(
        observations=observations,
        active_data_version_id=active_data_version_id,
    )

    if not history_log:
        history_log = "No prior executions. This is the first step—choose the first tool to call."

    deliverable_log = ""

    if deliverable_check:
        gate_type = deliverable_check.get("gate_type")

        if gate_type == "answer_quality_gate":
            deliverable_log += "Answer Quality Gate Feedback:\n"
        else:
            deliverable_log += "Deliverable Gate Feedback:\n"

        deliverable_log += f"- status: {deliverable_check.get('status')}\n"
        deliverable_log += f"- message: {deliverable_check.get('message')}\n"

        if gate_type == "answer_quality_gate":
            deliverable_log += f"- quality_status: {deliverable_check.get('quality_status')}\n"

            continuation_recommended = bool(
                deliverable_check.get("continuation_recommended")
            )
            available_evidence_categories = (
                    deliverable_check.get("available_evidence_categories", []) or []
            )
            covered_evidence_categories = (
                    deliverable_check.get("covered_evidence_categories", []) or []
            )
            missing_evidence_categories = (
                    deliverable_check.get("missing_evidence_categories", []) or []
            )
            missing_evidence_requirements = (
                    deliverable_check.get("missing_evidence_requirements", []) or []
            )

            if available_evidence_categories:
                deliverable_log += (
                        "- available_evidence_categories: "
                        + ", ".join(str(item) for item in available_evidence_categories)
                        + "\n"
                )

            if covered_evidence_categories:
                deliverable_log += (
                        "- covered_evidence_categories: "
                        + ", ".join(str(item) for item in covered_evidence_categories)
                        + "\n"
                )

            if missing_evidence_categories:
                deliverable_log += (
                        "- missing_evidence_categories: "
                        + ", ".join(str(item) for item in missing_evidence_categories)
                        + "\n"
                )

            if missing_evidence_requirements:
                deliverable_log += "- missing_evidence_requirements:\n"

                for item in missing_evidence_requirements:
                    deliverable_log += (
                        f"  - category: {item.get('evidence_category')}, "
                        f"required: {item.get('required_count')}, "
                        f"covered: {item.get('covered_count')}, "
                        f"missing: {item.get('missing_count')}\n"
                    )

            if continuation_recommended:
                deliverable_log += "\nCONTINUE_ANALYSIS_RECOMMENDED: true\n"
                deliverable_log += (
                    "The previous final answer did not yet satisfy the required evidence coverage. "
                    "Do not produce another final_answer unless the missing evidence is impossible to obtain. "
                    "Call exactly one appropriate analysis tool next. Use the available tool cards and their "
                    "plugin-declared evidence_categories to choose a tool that covers one missing evidence category.\n"
                )

            warnings = deliverable_check.get("warnings", []) or []
            if warnings:
                deliverable_log += "\nAnswer quality warnings:\n"
                for item in warnings:
                    deliverable_log += (
                        f"- check_id: {item.get('check_id')}\n"
                        f"  message: {item.get('message')}\n"
                        f"  recommendation: {item.get('recommendation')}\n"
                    )
        else:
            deliverable_log += f"- missing: {deliverable_check.get('missing')}\n"
            deliverable_log += f"- blocked: {deliverable_check.get('blocked')}\n"

            if deliverable_check.get("status") in {"missing", "blocked"}:
                deliverable_log += (
                    "\nCRITICAL: A previous final_answer was blocked by the DeliverableGate. "
                    "Do not produce final_answer yet. Call the tools needed to satisfy the missing deliverables, "
                    "unless the missing deliverable is truly unrecoverable.\n"
                )

        missing = deliverable_check.get("missing", []) or []
        if missing:
            deliverable_log += "\nMissing deliverables:\n"
            for item in missing:
                deliverable_log += (
                    f"- deliverable_id: {item.get('deliverable_id')}\n"
                    f"  description: {item.get('description')}\n"
                    f"  reason: {item.get('reason')}\n"
                    f"  satisfied_by: {item.get('satisfied_by')}\n"
                    f"  required_evidence: {item.get('required_evidence')}\n"
                    f"  missing_evidence: {item.get('missing_evidence')}\n"
                )

        blocked = deliverable_check.get("blocked", []) or []
        if blocked:
            deliverable_log += "\nBlocked deliverables:\n"
            for item in blocked:
                deliverable_log += f"- {item}\n"

        if deliverable_check.get("status") in {"missing", "blocked"}:
            deliverable_log += (
                "\nCRITICAL: A previous final_answer was blocked by the DeliverableGate. "
                "Do not produce final_answer yet. Call the tools needed to satisfy the missing deliverables, "
                "unless the missing deliverable is truly unrecoverable.\n"
            )

    data_version_log = ""

    if active_data_version_id:
        data_version_log += "\n### Active Data Version\n"
        data_version_log += f"- active_data_version_id: {active_data_version_id}\n"

        active_version = None
        for v in data_versions or []:
            if v.get("version_id") == active_data_version_id:
                active_version = v
                break

        if active_version:
            data_version_log += f"- rows: {active_version.get('n_rows')}\n"
            data_version_log += f"- columns: {active_version.get('n_cols')}\n"
            data_version_log += f"- operation: {active_version.get('operation')}\n"
            data_version_log += f"- parent_version_id: {active_version.get('parent_version_id')}\n"

    coverage_log = ""

    if analysis_coverage_brief:
        coverage_log += "\n### Analysis Coverage Brief\n"
        coverage_log += f"- analysis_goal: {analysis_coverage_brief.get('analysis_goal')}\n"
        coverage_log += (
                "- required_evidence_categories: "
                + ", ".join(analysis_coverage_brief.get("required_evidence_categories", []) or [])
                + "\n"
        )

        counts = analysis_coverage_brief.get("required_evidence_counts", {}) or {}
        if counts:
            coverage_log += f"- required_evidence_counts: {counts}\n"

        optional = analysis_coverage_brief.get("optional_evidence_categories", []) or []
        if optional:
            coverage_log += (
                    "- optional_evidence_categories: "
                    + ", ".join(optional)
                    + "\n"
            )

        coverage_log += f"- autonomy_level: {analysis_coverage_brief.get('autonomy_level')}\n"
        coverage_log += f"- reasoning_summary: {analysis_coverage_brief.get('reasoning_summary')}\n"

    context_text = (
        f"User request:\n{user_request}\n\n"
        f"Dataset overview:\n- rows: {rows}\n- columns: {cols}\n\n"
        f"{data_version_log}\n"
        f"{coverage_log}\n"
        f"History of actions and results:\n{history_log}\n\n"
        "Evidence reuse policy:\n"
        "- A previous observation may be reused for a numeric answer only if its "
        "data_version_id equals the current active_data_version_id.\n"
        "- Observations marked STALE must not be used to answer current numeric questions.\n"
        "- If the needed result is only available from a stale observation, call the appropriate tool again.\n\n"
        f"{deliverable_log}\n"
        "Read the history carefully. Do not repeat successful tools with the same intent. "
        "If you see an intervention warning, change strategy or output final_answer."
    )

    return ContextPackage(
        step=step,
        max_steps=max_steps,
        user_request=user_request,
        context_text=context_text,
        profile=profile,
        observations=observations,
        workspace_dir=workspace_dir,
        deliverable_check=deliverable_check,

        data_versions=data_versions or [],
        active_data_version_id=active_data_version_id,
        data_audit_log=data_audit_log or [],

    )
