import os
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from typing import TypedDict, Annotated
import operator


class RequiredDeliverable(BaseModel):
    deliverable_id: str = Field(..., description="Stable id for the deliverable.")
    description: str = Field(..., description="Human-readable deliverable description.")
    satisfied_by: List[str] = Field(default_factory=list, description="Tool names that can satisfy this deliverable.")
    required_evidence: List[str] = Field(default_factory=list, description="Evidence keys required to consider this deliverable complete.")
    status: Literal["pending", "satisfied", "missing", "blocked"] = "pending"


class TaskConstraint(BaseModel):
    constraint_id: str
    description: str
    type: Literal["data_mutation", "method", "reporting", "safety", "other"] = "other"


class TaskContract(BaseModel):
    contract_id: str = Field(..., description="Unique contract id.")
    user_goal: str = Field(..., description="What the user wants to accomplish.")
    required_deliverables: List[RequiredDeliverable] = Field(default_factory=list)
    constraints: List[TaskConstraint] = Field(default_factory=list)
    created_by: str = "supervisor"
    status: Literal["active", "satisfied", "blocked"] = "active"


class DeliverableCheckResult(BaseModel):
    status: Literal["ok", "missing", "blocked"] = "ok"
    satisfied: List[Dict[str, Any]] = Field(default_factory=list)
    missing: List[Dict[str, Any]] = Field(default_factory=list)
    blocked: List[Dict[str, Any]] = Field(default_factory=list)
    message: Optional[str] = None

# --- 1. User goals ---
class UserGoal(BaseModel):
    goal_id: str = Field(..., description="Unique goal ID")
    raw_request: str = Field(..., description="Original natural-language request")
    goal_type: Literal['explore_data', 'run_model', 'ask_clarification', 'unknown'] = Field(..., description="Goal category")

# --- 2. Data awareness ---
class ColumnProfile(BaseModel):
    name: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Data type")
    n_missing: int = Field(default=0, description="Missing value count")
    missing_rate: float = Field(default=0.0, description="Missing rate")
    n_unique: int = Field(default=0, description="Unique value count")
    semantic_type: str = Field(default="unknown", description="Coarse semantic type: numeric/categorical/datetime/boolean/text/unknown")
    is_numeric_like: bool = Field(default=False, description="Likely convertible to numeric")
    is_id_like: bool = Field(default=False, description="Likely an ID column")

class DatasetProfile(BaseModel):
    dataset_name: str = Field(..., description="Dataset name")
    n_rows: int = Field(..., description="Row count")
    n_cols: int = Field(..., description="Column count")
    columns: Dict[str, ColumnProfile] = Field(..., description="Per-column profile")


class GraphState(TypedDict):
    # observations must use operator.add for list merging
    observations: Annotated[list, operator.add]

    current_action: object
    current_execution: object
    current_step: int
    max_steps: int
    user_request: str
    workspace_dir: str

# --- 3. Actions and tools ---
class ToolSpec(BaseModel):
    name: str
    description: str
    func: Any  # Raw callable reference
    requires_confirmation: bool = False

    @classmethod
    def from_function(cls, func, requires_confirmation=False):
        """Build ToolSpec from function docstring and name."""
        return cls(
            name=func.__name__,
            description=func.__doc__ or "No description provided",
            func=func,
            requires_confirmation=requires_confirmation
        )

class ActionProposal(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    action_id: str = Field(..., description="Unique action ID")
    action_type: Literal['tool_call', 'ask_user', 'final_answer'] = Field(..., description="Action type")
    tool_name: Optional[str] = Field(None, description="Tool name when tool_call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    reasoning_summary: str = Field(..., description="Reasoning for this action")
    task_contract: Optional[TaskContract] = None
    contract_update: Optional[Dict[str, Any]] = None

# --- 4. Verification and execution ---
class VerificationResult(BaseModel):
    action_id: str = Field(..., description="Related action ID")
    status: Literal['allowed', 'rejected_recoverable', 'rejected_terminal', 'needs_review'] = Field(..., description="Verification status")
    feedback: Optional[str] = Field(None, description="Feedback for the LLM")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured verification details")

class ToolExecutionResult(BaseModel):
    execution_id: str = Field(..., description="Unique execution record ID")
    action_id: str = Field(..., description="Related action ID")
    tool_name: Optional[str] = Field(None, description="Tool name")

    # success is a legacy boolean summary for UI; status is the semantic outcome.
    success: bool = Field(..., description="Whether the tool task succeeded semantically")
    status: Literal["ok", "warning", "blocked", "failed"] = Field(default="ok", description="Semantic execution status")

    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    message: Optional[str] = Field(None, description="Human-readable message")
    recoverable: bool = Field(default=False, description="Whether recovery is possible")

    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured tool return data")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Generated files, plots, etc.")

class Observation(BaseModel):
    observation_id: str = Field(..., description="Unique observation ID")
    source_action_id: str = Field(..., description="Source action ID")
    tool_name: Optional[str] = Field(None, description="Source tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    status: Literal["ok", "warning", "blocked", "failed", "rejected"] = Field(default="ok")
    success: bool = Field(default=True)
    error_code: Optional[str] = None
    message: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)

    summary: str = Field(..., description="Short summary for the LLM")
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="Structured metrics")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw execution payload")

# --- 5. Context packaging ---
class ContextPackage(BaseModel):
    context_text: str
    current_step: int
    max_steps: int
    workspace_dir: str = "./"

class AgentContext:
    def __init__(self, workspace_dir: str, arguments: dict = None):
        self.workspace_dir = workspace_dir
        self.args = arguments or {}
        self.file_path = os.path.join(self.workspace_dir, "working_data.parquet")

    def _get_data_path(self):
        """Resolve a tabular data file inside the sandbox."""
        for file in os.listdir(self.workspace_dir):
            if file.endswith(('.csv', '.xls', '.xlsx')):
                return os.path.join(self.workspace_dir, file)
        raise FileNotFoundError(f"No supported tabular file found in sandbox {self.workspace_dir}")

    def load_df(self):
        """Load dataframe from sandbox Parquet."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found in sandbox: {self.file_path}")
        return pd.read_parquet(self.file_path, engine='pyarrow')

    def save_df(self, df):
        """Write dataframe back to sandbox Parquet."""
        df.to_parquet(self.file_path, engine='pyarrow', index=False)

    def get_arg(self, key: str, default=None):
        return self.args.get(key, default)

