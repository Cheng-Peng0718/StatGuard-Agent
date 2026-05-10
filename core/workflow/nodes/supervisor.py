from __future__ import annotations

from core.action_access import (
    get_action_field,
    get_action_reasoning_summary,
    get_action_type,
)
from core.context_builder import build_context
from core.services.task_contracts import task_contract_to_state_dict
from core.workflow.profile_access import get_context_profile

def call_supervisor(context_pkg):
    from agents.supervisor import call_supervisor as real_call_supervisor

    return real_call_supervisor(context_pkg)


def supervisor_node(state: dict):
    current_workspace = state.get("workspace_dir", "./")
    current_profile = get_context_profile(state)

    context_pkg = build_context(
        step=state.get("current_step", 1),
        max_steps=state.get("max_steps", 12),
        user_request=state.get("user_request", "Not provided"),
        profile=current_profile,
        observations=state.get("observations", []),
        workspace_dir=current_workspace,
        deliverable_check=state.get("deliverable_check"),
        data_versions=state.get("data_versions", []),
        active_data_version_id=state.get("active_data_version_id"),
        data_audit_log=state.get("data_audit_log", []),
        task_contract=state.get("task_contract"),
    )

    action = call_supervisor(context_pkg)
    updates = {"current_action": action}

    print("\n" + "=" * 40)
    print(f"[Supervisor decision]: action_type = {get_action_type(action)}")
    print(f"[Reasoning summary]: {get_action_reasoning_summary(action)}")
    print("=" * 40 + "\n")

    contract = get_action_field(action, "task_contract", None)
    contract_dict = task_contract_to_state_dict(contract)

    if contract_dict:
        print(
            f"[TASK CONTRACT DECLARED] "
            f"deliverables={len(contract_dict.get('required_deliverables', []))}"
        )

        updates["task_contract"] = contract_dict

    return updates
