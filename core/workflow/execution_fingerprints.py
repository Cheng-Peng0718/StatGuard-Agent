from __future__ import annotations

from typing import Any, Iterable

from core.workflow.runtime_utils import get_action_hash


NON_EXECUTION_ERROR_CODES = {
    "HUMAN_CONFIRMATION_REQUIRED",
    "HUMAN_REVIEW_REJECTED",
    "VERIFICATION_FAILED",
    "MISSING_REVIEW_STATE",
    "UNHANDLED_HUMAN_REVIEW_STATUS",
}


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    return {}


def _looks_like_real_execution_observation(obs: dict) -> bool:
    """
    Return True only for observations produced from actual tool execution.

    Important:
    - Human review requests are not executions.
    - Verification rejections are not executions.
    - Missing review state / fallback review errors are not executions.
    """
    if not isinstance(obs, dict):
        return False

    if not obs.get("tool_name"):
        return False

    error_code = obs.get("error_code")
    if error_code in NON_EXECUTION_ERROR_CODES:
        return False

    raw_data = obs.get("raw_data") or {}
    structured_data = obs.get("structured_data") or {}

    if isinstance(raw_data, dict) and raw_data.get("execution_id"):
        return True

    if isinstance(structured_data, dict):
        payload = structured_data.get("payload") or {}
        if isinstance(payload, dict) and payload:
            return True

    # Conservative fallback:
    # summarize_node-created real tool observations normally have success/status.
    # Review/verification observations are filtered above by error_code.
    return obs.get("success") is not None and obs.get("status") in {
        "ok",
        "warning",
        "failed",
        "blocked",
    }


def iter_executed_action_hashes(state: dict) -> Iterable[str]:
    """
    Yield hashes for actions that represent real tool execution attempts.

    Prefer analysis_runs as the canonical execution ledger.
    Fall back to execution-looking observations for older states.
    """
    seen = set()

    for run in state.get("analysis_runs", []) or []:
        run_dict = _as_dict(run)
        tool_name = run_dict.get("tool_name")
        arguments = run_dict.get("arguments") or {}

        if not tool_name:
            continue

        action_hash = get_action_hash(tool_name, arguments)
        seen.add(action_hash)
        yield action_hash

    for obs in state.get("observations", []) or []:
        obs_dict = _as_dict(obs)

        if not _looks_like_real_execution_observation(obs_dict):
            continue

        tool_name = obs_dict.get("tool_name")
        arguments = obs_dict.get("arguments") or {}

        action_hash = get_action_hash(tool_name, arguments)

        if action_hash in seen:
            continue

        seen.add(action_hash)
        yield action_hash


def has_duplicate_executed_action(
    *,
    state: dict,
    tool_name: str,
    arguments: dict,
) -> bool:
    current_hash = get_action_hash(tool_name, arguments or {})
    return current_hash in set(iter_executed_action_hashes(state))