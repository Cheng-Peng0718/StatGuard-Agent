from __future__ import annotations

from typing import Any, Dict, List, Set


def as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def get_state_value(state: Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


def get_action_field(action: Any, field_name: str, default=None):
    if action is None:
        return default

    if isinstance(action, dict):
        return action.get(field_name, default)

    return getattr(action, field_name, default)


def normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [str(v) for v in value if v is not None]

    if isinstance(value, tuple):
        return [str(v) for v in value if v is not None]

    if isinstance(value, set):
        return [str(v) for v in value if v is not None]

    return []


def extract_final_answer_content_from_state(state: Any) -> str | None:
    """
    Extract final-answer text from backend graph state.

    This supports the current supervisor final-answer bridge:
    - state["final_answer"]
    - state["current_action"].final_answer / answer / content / message
    - state["current_action"].arguments["final_answer" / "answer" / "content" / "message"]
    """
    direct = get_state_value(state, "final_answer")

    if isinstance(direct, str) and direct.strip():
        return direct

    action = get_state_value(state, "current_action")

    for field_name in ["final_answer", "answer", "content", "message"]:
        value = get_action_field(action, field_name)

        if isinstance(value, str) and value.strip():
            return value

    arguments = get_action_field(action, "arguments", {}) or {}

    if isinstance(arguments, dict):
        for key in ["final_answer", "answer", "content", "message"]:
            value = arguments.get(key)

            if isinstance(value, str) and value.strip():
                return value

    return None


def get_deliverable_evidence(state: Any) -> Dict[str, Any]:
    """
    Return explicit deliverable evidence stored in state.

    Expected shape:
    {
        "satisfied_deliverables": [...],
        "satisfied_criteria": [...],
        "satisfied": [...]
    }
    """
    return as_dict(get_state_value(state, "deliverable_evidence", {}))


def get_satisfied_deliverable_names(state: Any) -> Set[str]:
    evidence = get_deliverable_evidence(state)

    names = set(normalize_string_list(evidence.get("satisfied_deliverables")))

    for item in normalize_string_list(evidence.get("satisfied")):
        if item.startswith("deliverable:"):
            names.add(item.replace("deliverable:", "", 1))

    return names


def get_satisfied_criterion_names(state: Any) -> Set[str]:
    evidence = get_deliverable_evidence(state)

    names = set(normalize_string_list(evidence.get("satisfied_criteria")))

    for item in normalize_string_list(evidence.get("satisfied")):
        if item.startswith("criterion:"):
            names.add(item.replace("criterion:", "", 1))

    return names


def criterion_satisfied_by_final_answer_text(
    criterion: str,
    final_answer_text: str | None,
) -> bool:
    """
    Deterministic criterion matching.

    Supported syntax:
    - "contains:<phrase>"

    Free-form criteria are not automatically considered satisfied in S10C.
    """
    if not final_answer_text:
        return False

    if not criterion.startswith("contains:"):
        return False

    phrase = criterion.replace("contains:", "", 1).strip()

    if not phrase:
        return False

    return phrase.lower() in final_answer_text.lower()