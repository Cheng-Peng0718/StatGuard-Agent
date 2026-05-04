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