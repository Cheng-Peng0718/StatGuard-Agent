from __future__ import annotations

from typing import Any, Dict

from core.planning.schemas import PlanProposal
from core.services.intelligent_planner import create_plan_from_state


def create_llm_plan_from_state(state: Dict[str, Any]) -> PlanProposal:
    """
    LLM-first planner service boundary.

    Temporary implementation:
    - delegates to the deterministic planner as a compatibility fallback;
    - keeps active workflow nodes decoupled from intelligent_planner;
    - future implementation will call an LLM using DatasetContext and ToolManifest registry.
    """
    return create_plan_from_state(state)