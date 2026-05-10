from __future__ import annotations

from core.responses import make_response_update
from core.services.llm_interaction_parser import (
    decide_llm_interaction_intent,
    legacy_interaction_intent_from_decision,
)

def intent_router_node(state: dict):
    user_request = state.get("user_request", "")
    decision = decide_llm_interaction_intent(user_request, state=state)
    legacy_intent = legacy_interaction_intent_from_decision(decision)

    print("\n" + "=" * 40)
    print("[INTENT ROUTER]")
    print(f"user_request = {user_request}")
    print(f"intent = {decision.intent}")
    print(f"legacy_intent = {legacy_intent}")
    print("=" * 40 + "\n")

    updates = {
        "interaction_intent": legacy_intent,
        "intent_decision": decision.model_dump(),
    }

    if decision.task_spec is not None:
        updates["task_spec"] = decision.task_spec.model_dump()

    return updates


def advisory_answer_node(state: dict):
    summary = state.get("dataset_summary") or {}
    capability_map = state.get("capability_map") or {}

    n_rows = summary.get("n_rows", "unknown")
    n_cols = summary.get("n_cols", "unknown")

    numeric_cols = summary.get("numeric_columns", []) or []
    categorical_cols = summary.get("categorical_columns", []) or []
    binary_cols = summary.get("binary_columns", []) or []
    id_like_cols = summary.get("id_like_columns", []) or []
    missingness = summary.get("missingness_summary", {}) or {}

    capabilities = capability_map.get("capabilities", []) or []

    ready = [c for c in capabilities if c.get("status") == "ready"]
    needs_choice = [c for c in capabilities if c.get("status") == "needs_user_choice"]
    not_applicable = [
        c for c in capabilities
        if c.get("status") in {"not_applicable", "blocked"}
    ]

    lines = []

    lines.append("I have profiled the current dataset. Here is what you can do next.")
    lines.append("")
    lines.append("Dataset overview:")
    lines.append(f"- Rows: {n_rows}")
    lines.append(f"- Columns: {n_cols}")
    lines.append(f"- Numeric columns: {len(numeric_cols)}")
    lines.append(f"- Categorical columns: {len(categorical_cols)}")
    lines.append(f"- Binary columns: {len(binary_cols)}")
    lines.append(f"- ID-like columns: {len(id_like_cols)}")
    lines.append(
        f"- Columns with missing values: "
        f"{missingness.get('n_columns_with_missing', 0)}"
    )
    lines.append("")

    if ready:
        lines.append("Analyses that appear ready:")
        for cap in ready[:8]:
            lines.append(
                f"- {cap.get('display_name', cap.get('tool_name'))}: "
                f"{cap.get('reason')}"
            )
        lines.append("")

    if needs_choice:
        lines.append("Analyses that may be useful but need your choices first:")
        for cap in needs_choice[:8]:
            choices = ", ".join(cap.get("required_user_choices", []) or [])
            lines.append(
                f"- {cap.get('display_name', cap.get('tool_name'))}: "
                f"needs {choices or 'additional choices'}"
            )
        lines.append("")

    if not_applicable:
        lines.append("Currently blocked or not recommended:")
        for cap in not_applicable[:5]:
            lines.append(
                f"- {cap.get('display_name', cap.get('tool_name'))}: "
                f"{cap.get('reason')}"
            )
        lines.append("")

    lines.append("I have not run any analysis tools yet.")
    lines.append(
        "If you want, say `make a plan` and I will draft a data-aware plan without executing it."
    )

    answer = "\n".join(lines)

    updates = make_response_update(
        response_type="advisory",
        content=answer,
        source_node="advisory_answer",
        data_version_id=state.get("active_data_version_id"),
        metadata={
            "interaction_intent": state.get("interaction_intent"),
        },
    )

    updates.update({
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
    })

    return updates
