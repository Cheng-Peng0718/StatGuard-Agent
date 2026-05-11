from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.analysis_tool_plugins import get_plugin
from core.repair.proposal import (
    make_argument_repair_proposal,
    make_ask_user_repair_proposal,
    make_no_op_repair_proposal,
)


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


def _get_field(value: Any, field_name: str, default=None):
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def _get_action_arguments(current_action: Any) -> Dict[str, Any]:
    args = _get_field(current_action, "arguments", {}) or {}

    if isinstance(args, dict):
        return dict(args)

    return {}


def _get_argument_schema(plugin: Any):
    if plugin is None:
        return None

    return getattr(plugin, "argument_schema", None)


def _normalize_string_value_with_aliases(
    *,
    field_name: str,
    value: Any,
    value_aliases: Dict[str, Dict[str, Any]],
) -> Tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False

    aliases_for_field = value_aliases.get(field_name, {}) or {}

    if not isinstance(aliases_for_field, dict):
        return value, False

    raw = value.strip()
    lowered = raw.lower()

    if raw in aliases_for_field:
        return aliases_for_field[raw], True

    if lowered in aliases_for_field:
        return aliases_for_field[lowered], True

    return value, False


def _apply_schema_value_aliases(
    *,
    arguments: Dict[str, Any],
    argument_schema: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generic deterministic argument repair using plugin ArgumentSchema.value_aliases.

    Example:
    strategy="drop" -> strategy="rows"
    if the plugin schema declares that alias.

    Returns:
    - repaired_arguments
    - change_report
    """
    value_aliases = getattr(argument_schema, "value_aliases", {}) or {}

    if not isinstance(value_aliases, dict):
        return dict(arguments), {}

    repaired = dict(arguments)
    changes = {}

    for field_name, value in arguments.items():
        new_value, changed = _normalize_string_value_with_aliases(
            field_name=field_name,
            value=value,
            value_aliases=value_aliases,
        )

        if changed:
            repaired[field_name] = new_value
            changes[field_name] = {
                "old": value,
                "new": new_value,
                "repair": "value_alias",
            }

    return repaired, changes


def _find_missing_required_fields(
    *,
    arguments: Dict[str, Any],
    argument_schema: Any,
) -> list[str]:
    required = getattr(argument_schema, "required", {}) or {}

    if not isinstance(required, dict):
        return []

    missing = []

    for field_name in required.keys():
        value = arguments.get(field_name)

        if value is None or value == "":
            missing.append(field_name)

    return missing


def generate_repair_proposal(
    *,
    repair_decision: Any,
    current_action: Any,
) -> Dict[str, Any]:
    """
    Deterministically generate a repair proposal.

    This function does not execute tools, does not retry, and does not call LLMs.

    Current deterministic capabilities:
    1. terminal/no_repair_needed -> no_op proposal
    2. needs_user -> ask_user proposal
    3. repairable + schema value_aliases -> argument_repair proposal
    4. repairable but no deterministic fix -> no_op proposal
    """
    decision = _as_dict(repair_decision)

    decision_status = decision.get("status")
    tool_name = decision.get("tool_name") or _get_field(current_action, "tool_name")
    error_code = decision.get("error_code")

    if decision_status in {None, "no_repair_needed"}:
        return make_no_op_repair_proposal(
            repair_decision=decision,
            current_action=current_action,
            reason="No repair is needed.",
        )

    if decision_status == "terminal":
        return make_no_op_repair_proposal(
            repair_decision=decision,
            current_action=current_action,
            reason="Repair decision is terminal; no deterministic repair proposal is allowed.",
        )

    plugin = get_plugin(tool_name)

    if plugin is None:
        return make_no_op_repair_proposal(
            repair_decision=decision,
            current_action=current_action,
            reason="No plugin contract is available for deterministic repair.",
        )

    argument_schema = _get_argument_schema(plugin)
    arguments = _get_action_arguments(current_action)

    if argument_schema is None:
        return make_no_op_repair_proposal(
            repair_decision=decision,
            current_action=current_action,
            reason="Plugin has no argument schema available for deterministic repair.",
        )

    missing_required = _find_missing_required_fields(
        arguments=arguments,
        argument_schema=argument_schema,
    )

    if decision_status == "needs_user" or error_code in {
        "MISSING_REQUIRED_ROLE",
        "MISSING_USER_CHOICE",
        "MISSING_COLUMNS",
    }:
        missing_fields = missing_required or ["user_choice"]

        return make_ask_user_repair_proposal(
            repair_decision=decision,
            current_action=current_action,
            prompt=(
                "Repair requires user-provided choices before the action can be retried."
            ),
            missing_fields=missing_fields,
        )

    if decision_status == "repairable":
        repaired_arguments, changes = _apply_schema_value_aliases(
            arguments=arguments,
            argument_schema=argument_schema,
        )

        if changes:
            return make_argument_repair_proposal(
                repair_decision=decision,
                current_action=current_action,
                proposed_arguments=repaired_arguments,
                reason=(
                    "Applied deterministic argument normalization using the plugin "
                    "argument schema value aliases."
                ),
                requires_confirmation=getattr(plugin, "requires_confirmation", False),
                risk_level="medium" if getattr(plugin, "requires_confirmation", False) else "low",
            )

        return make_no_op_repair_proposal(
            repair_decision=decision,
            current_action=current_action,
            reason=(
                "The failure is repairable in principle, but no deterministic "
                "argument repair could be inferred from the plugin schema."
            ),
        )

    return make_no_op_repair_proposal(
        repair_decision=decision,
        current_action=current_action,
        reason=f"Unsupported repair decision status: {decision_status}.",
    )