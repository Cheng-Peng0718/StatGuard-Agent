from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from core.analysis_tool_plugins import get_plugin


class RepairDecision(BaseModel):
    status: Literal[
        "no_repair_needed",
        "repairable",
        "needs_user",
        "terminal",
    ]
    reason: str

    tool_name: Optional[str] = None
    error_code: Optional[str] = None

    allow_argument_repair: bool = False
    allow_method_fallback: bool = False
    requires_user_for_missing_roles: bool = False

    evidence: Dict[str, Any] = Field(default_factory=dict)


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def _get_state_value(state: Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


def _get_action_tool_name(action: Any) -> Optional[str]:
    if action is None:
        return None

    if isinstance(action, dict):
        return action.get("tool_name")

    return getattr(action, "tool_name", None)


def _get_error_code_from_verification(verification: Any) -> Optional[str]:
    verification_dict = _as_dict(verification)

    return (
        verification_dict.get("error_code")
        or verification_dict.get("code")
    )


def _get_error_code_from_execution(execution: Any) -> Optional[str]:
    execution_dict = _as_dict(execution)

    return (
        execution_dict.get("error_code")
        or execution_dict.get("code")
    )


def _get_status(value: Any) -> Optional[str]:
    value_dict = _as_dict(value)

    return value_dict.get("status")


def _get_repair_policy(tool_name: Optional[str]):
    if not tool_name:
        return None

    plugin = get_plugin(tool_name)

    if plugin is None:
        return None

    return getattr(plugin, "repair_policy", None)


def evaluate_repair_decision(state: Any) -> RepairDecision:
    """
    Decide whether the current failed verification/execution is repairable.

    This function does not repair, retry, call tools, or call LLMs.
    It only classifies the next backend action.
    """
    current_action = _get_state_value(state, "current_action")
    current_verification = _get_state_value(state, "current_verification")
    current_execution = _get_state_value(state, "current_execution")

    tool_name = _get_action_tool_name(current_action)

    verification_status = _get_status(current_verification)
    execution_status = _get_status(current_execution)

    verification_error_code = _get_error_code_from_verification(current_verification)
    execution_error_code = _get_error_code_from_execution(current_execution)

    error_code = verification_error_code or execution_error_code

    evidence = {
        "verification_status": verification_status,
        "execution_status": execution_status,
        "verification_error_code": verification_error_code,
        "execution_error_code": execution_error_code,
    }

    if verification_status not in {"rejected_recoverable", "rejected_terminal"} and execution_status not in {"failed", "error", "rejected"}:
        return RepairDecision(
            status="no_repair_needed",
            reason="No failed verification or failed execution is present.",
            tool_name=tool_name,
            error_code=error_code,
            evidence=evidence,
        )

    if verification_status == "rejected_terminal":
        return RepairDecision(
            status="terminal",
            reason="Verification rejected the action as terminal.",
            tool_name=tool_name,
            error_code=error_code,
            evidence=evidence,
        )

    repair_policy = _get_repair_policy(tool_name)

    if repair_policy is None:
        return RepairDecision(
            status="terminal",
            reason="No repair policy is available for this tool.",
            tool_name=tool_name,
            error_code=error_code,
            evidence=evidence,
        )

    non_repairable = set(getattr(repair_policy, "non_repairable_error_codes", []) or [])
    repairable = set(getattr(repair_policy, "repairable_error_codes", []) or [])

    if error_code in non_repairable:
        return RepairDecision(
            status="terminal",
            reason=f"Error code {error_code} is marked non-repairable.",
            tool_name=tool_name,
            error_code=error_code,
            allow_argument_repair=getattr(repair_policy, "allow_argument_repair", False),
            allow_method_fallback=getattr(repair_policy, "allow_method_fallback", False),
            requires_user_for_missing_roles=getattr(repair_policy, "requires_user_for_missing_roles", False),
            evidence=evidence,
        )

    if getattr(repair_policy, "requires_user_for_missing_roles", False) and error_code in {
        "MISSING_REQUIRED_ROLE",
        "MISSING_USER_CHOICE",
        "MISSING_COLUMNS",
    }:
        return RepairDecision(
            status="needs_user",
            reason="Repair requires user-provided roles or choices.",
            tool_name=tool_name,
            error_code=error_code,
            allow_argument_repair=getattr(repair_policy, "allow_argument_repair", False),
            allow_method_fallback=getattr(repair_policy, "allow_method_fallback", False),
            requires_user_for_missing_roles=True,
            evidence=evidence,
        )

    if error_code in repairable or verification_status == "rejected_recoverable":
        return RepairDecision(
            status="repairable",
            reason="The error is recoverable under the tool repair policy.",
            tool_name=tool_name,
            error_code=error_code,
            allow_argument_repair=getattr(repair_policy, "allow_argument_repair", False),
            allow_method_fallback=getattr(repair_policy, "allow_method_fallback", False),
            requires_user_for_missing_roles=getattr(repair_policy, "requires_user_for_missing_roles", False),
            evidence=evidence,
        )

    return RepairDecision(
        status="terminal",
        reason="The failure is not listed as repairable.",
        tool_name=tool_name,
        error_code=error_code,
        allow_argument_repair=getattr(repair_policy, "allow_argument_repair", False),
        allow_method_fallback=getattr(repair_policy, "allow_method_fallback", False),
        requires_user_for_missing_roles=getattr(repair_policy, "requires_user_for_missing_roles", False),
        evidence=evidence,
    )