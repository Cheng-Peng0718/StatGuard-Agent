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
    def load_df(self):
        if self.active_data_version_id and self.data_versions:
            for version in self.data_versions:
                if version.get("version_id") == self.active_data_version_id:
                    path = version.get("path")
                    if path and os.path.exists(path):

                        ##### DEBUG
                        print(f"[LOAD DF] active_version={self.active_data_version_id}, path={path}")
                        ##### DEBUG

                        return pd.read_parquet(path)

        fallback_path = os.path.join(self.workspace_dir, "working_data.parquet")
        if os.path.exists(fallback_path):

            ##### DEBUG
            print(f"[LOAD DF] fallback working_data, path={fallback_path}")
            ##### DEBUG

            return pd.read_parquet(fallback_path)



        raise FileNotFoundError(
            f"No active data version or fallback working_data.parquet found in {self.workspace_dir}"
        )

    def active_data_path(self):

        if self.active_data_version_id and self.data_versions:
            for version in self.data_versions:
                if version.get("version_id") == self.active_data_version_id:
                    path = version.get("path")
                    if path:
                        return path

        return os.path.join(self.workspace_dir, "working_data.parquet")

    step: int
    max_steps: int
    user_request: str
    profile: Any = None
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    workspace_dir: str = "./"
    context_text: str = ""

    data_versions: List[Dict[str, Any]] = Field(default_factory=list)
    active_data_version_id: Optional[str] = None
    data_audit_log: List[Dict[str, Any]] = Field(default_factory=list)

class AgentContext:
    def __init__(
        self,
        workspace_dir: str,
        arguments: dict = None,
        data_versions: list = None,
        active_data_version_id: str = None,
        data_audit_log: list = None,
    ):
        self.workspace_dir = workspace_dir
        self.args = arguments or {}

        # Backward-compatible fallback path
        self.file_path = os.path.join(self.workspace_dir, "working_data.parquet")

        # Phase 2: dataset versioning
        self.data_versions = data_versions or []
        self.active_data_version_id = active_data_version_id
        self.data_audit_log = data_audit_log or []

    def _get_data_path(self):
        """Resolve a tabular data file inside the sandbox."""
        for file in os.listdir(self.workspace_dir):
            if file.endswith(('.csv', '.xls', '.xlsx')):
                return os.path.join(self.workspace_dir, file)
        raise FileNotFoundError(f"No supported tabular file found in sandbox {self.workspace_dir}")

    def active_data_path(self):
        """
        Resolve the active data version path.

        Priority:
        1. active_data_version_id from data_versions
        2. backward-compatible working_data.parquet
        """
        if self.active_data_version_id and self.data_versions:
            for version in self.data_versions:
                if version.get("version_id") == self.active_data_version_id:
                    path = version.get("path")
                    if path:
                        return path

        return self.file_path

    def load_df(self):
        """
        Load dataframe from the active data version if available.
        Fall back to working_data.parquet for backward compatibility.
        """
        path = self.active_data_path()

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found in sandbox: {path}")

        if self.active_data_version_id:
            print(f"[LOAD DF] active_version={self.active_data_version_id}, path={path}")
        else:
            print(f"[LOAD DF] fallback working_data, path={path}")

        return pd.read_parquet(path, engine='pyarrow')

    def save_df(self, df):
        """
        Backward-compatible save.

        Warning:
        Versioned tools such as clean_data should not use this directly.
        They should create a new data version instead.
        """
        df.to_parquet(self.file_path, engine='pyarrow', index=False)

    def get_arg(self, key: str, default=None):
        return self.args.get(key, default)

class DataVersion(BaseModel):
    version_id: str = Field(..., description="Stable data version id, e.g. raw_v1 or data_v0002.")
    parent_version_id: Optional[str] = Field(default=None, description="Parent data version.")
    path: str = Field(..., description="Parquet file path for this version.")
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    created_by: str = "system"
    created_at: Optional[str] = None
    operation: str = "initial_load"
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataAuditEvent(BaseModel):
    event_id: str
    event_type: Literal[
        "data_loaded",
        "data_cleaned",
        "data_version_created",
        "data_version_activated",
        "data_version_rolled_back",
        "tool_used_data_version",
    ]
    version_id: Optional[str] = None
    parent_version_id: Optional[str] = None
    tool_name: Optional[str] = None
    action_id: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None

class GuardrailFinding(BaseModel):
    """
    A structured statistical guardrail finding attached to an analysis run.

    severity:
    - info: useful context
    - warning: potential issue
    - critical: serious issue that may invalidate interpretation
    """
    finding_id: str
    category: str
    severity: str = "info"
    title: str
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None

class AnalysisRun(BaseModel):
    """
    A structured record of one completed analysis/tool run.

    This is used by the UI Results Panel and future report export.
    """
    run_id: str
    tool_name: str
    action_id: Optional[str] = None
    data_version_id: Optional[str] = None

    status: str = "unknown"
    success: bool = False
    created_at: Optional[str] = None

    title: str = ""
    summary: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)

    metrics: Dict[str, Any] = Field(default_factory=dict)
    tables: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)

    report_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    guardrails: List[Dict[str, Any]] = Field(default_factory=list)

    raw_observation_id: Optional[str] = None



