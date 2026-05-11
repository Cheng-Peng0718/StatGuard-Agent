from __future__ import annotations

from core.dataset_intelligence.schemas import CapabilityMap, DatasetProfileV2
from core.planning.planner import build_plan_from_capability_map
from core.planning.renderer import render_plan_for_user
from core.planning.verifier import verify_plan
from core.responses import make_response_update


def plan_only_node(state: dict):
    capability_map_dict = state.get("capability_map")
    profile_dict = state.get("dataset_profile_v2")

    if not capability_map_dict or not profile_dict:
        content = (
            "I cannot create a data-aware plan yet because the dataset profile "
            "is not available. Please upload or reload a dataset first."
        )

        updates = make_response_update(
            response_type="error",
            content=content,
            source_node="plan_only",
            data_version_id=state.get("active_data_version_id"),
            metadata={
                "reason": "missing_dataset_profile_or_capability_map",
            },
        )

        updates.update({
            "pending_plan": None,
            "plan_status": "blocked",
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
        })

        return updates

    capability_map = CapabilityMap.model_validate(capability_map_dict)
    dataset_profile = DatasetProfileV2.model_validate(profile_dict)

    plan = build_plan_from_capability_map(
        user_request=state.get("user_request", ""),
        capability_map=capability_map,
    )

    verified_plan = verify_plan(plan, dataset_profile)
    rendered = render_plan_for_user(verified_plan)

    print("\n" + "=" * 40)
    print("[PLAN ONLY NODE]")
    print(f"plan_id = {verified_plan.plan_id}")
    print(f"plan_status = {verified_plan.status}")
    print(f"n_steps = {len(verified_plan.steps)}")
    print(f"n_blocked = {len(verified_plan.blocked_or_not_recommended)}")
    print("=" * 40 + "\n")

    updates = make_response_update(
        response_type="plan",
        content=rendered,
        source_node="plan_only",
        data_version_id=state.get("active_data_version_id"),
        plan_id=verified_plan.plan_id,
        plan_status=verified_plan.status,
        metadata={
            "interaction_intent": state.get("interaction_intent"),
            "n_steps": len(verified_plan.steps),
            "n_blocked": len(verified_plan.blocked_or_not_recommended),
        },
    )

    updates.update({
        "pending_plan": verified_plan.model_dump(),
        "plan_status": verified_plan.status,

        # Hard safety reset.
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,
    })

    return updates