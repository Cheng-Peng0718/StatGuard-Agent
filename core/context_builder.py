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

        lines.append(
            f"- tool={tool_name}, status={status}, success={success}, "
            f"data_version_id={obs_version}, version_status={version_status}, "
            f"message={message}, summary={summary}"
        )

        # 关键：只有 CURRENT observation 才允许暴露 payload
        if version_status == "CURRENT":
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
            if pd.api.types.is_categorical_dtype(s):
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
        deliverable_log += "Deliverable Gate Feedback:\n"
        deliverable_log += f"- status: {deliverable_check.get('status')}\n"
        deliverable_log += f"- message: {deliverable_check.get('message')}\n"

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

    context_text = (
        f"User request:\n{user_request}\n\n"
        f"Dataset overview:\n- rows: {rows}\n- columns: {cols}\n\n"
        f"{data_version_log}\n"
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
