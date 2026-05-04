import os

import pandas as pd
from core.schema import ContextPackage, DatasetProfile, ColumnProfile
from tools.registry import registry


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


def build_context(step, max_steps, user_request, profile, observations, workspace_dir="./", deliverable_check=None):
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

    history_log = ""
    for idx, obs in enumerate(observations):
        if isinstance(obs, dict):
            t_name = obs.get("tool_name", "unknown_tool")
            status = obs.get("status") or obs.get("structured_data", {}).get("status", "unknown")
            success = obs.get("success") if "success" in obs else obs.get("structured_data", {}).get("success")
            error_code = obs.get("error_code") or obs.get("structured_data", {}).get("error_code")
            message = obs.get("message") or obs.get("summary", "")
            artifacts = obs.get("artifacts") or obs.get("structured_data", {}).get("artifacts", [])

            marker = "✅" if status in ["ok", "warning"] else "❌"
            history_log += (
                f"{marker} [step {idx + 1}] tool: {t_name}\n"
                f"- status: {status}\n"
                f"- success: {success}\n"
            )

            if error_code:
                history_log += f"- error_code: {error_code}\n"

            if message:
                history_log += f"- message: {str(message)[:500]}\n"

            if artifacts:
                history_log += f"- artifacts: {artifacts}\n"

            payload = obs.get("structured_data", {}).get("payload")
            if payload:
                history_log += f"- key payload: {str(payload)[:800]}\n"

            history_log += "\n"

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

    context_text = (
        f"User request:\n{user_request}\n\n"
        f"Dataset overview:\n- rows: {rows}\n- columns: {cols}\n\n"
        f"History of actions and results:\n{history_log}\n\n"
        f"{deliverable_log}\n"
        f"Read the history carefully. Do not repeat successful tools with the same intent. "
        f"If you see a Deliverable Gate warning, satisfy the missing deliverables before final_answer. "
        f"If you see an intervention warning, change strategy or output final_answer."
    )

    return ContextPackage(
        context_text=context_text,
        current_step=step,
        max_steps=max_steps,
        workspace_dir=workspace_dir
    )
