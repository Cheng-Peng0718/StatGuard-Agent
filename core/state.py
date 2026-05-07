import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from core.schema import ActionProposal, DatasetProfile, ToolExecutionResult, VerificationResult


class GraphState(TypedDict):
    user_request: str
    dataset_profile: Any

    workspace_dir: str

    observations: Annotated[list, operator.add]

    current_action: Any
    current_execution: Any
    current_verification: Any
    current_step: int
    max_steps: int
    user_request: str
    workspace_dir: str
    task_contract: Optional[Dict[str, Any]]
    deliverable_check: Optional[Dict[str, Any]]
    deliverable_gate_attempts: int
    data_versions: Optional[List[Dict[str, Any]]]
    active_data_version_id: Optional[str]
    data_audit_log: Optional[List[Dict[str, Any]]]
    analysis_runs: Optional[List[Dict[str, Any]]]
    action_origin: str

    interaction_intent: str
    dataset_profile_v2: dict
    dataset_summary: dict
    capability_map: dict
    pending_plan: dict
    plan_status: str
    final_answer: str
    assistant_response: dict
    execution_audit: dict
    repair_decision: dict
    repair_attempts: list
    repair_proposal: dict
    state_serialization_audit: dict