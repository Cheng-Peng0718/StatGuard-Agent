from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExecutionAuditIssue(BaseModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    location: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ExecutionAuditResult(BaseModel):
    status: Literal["ok", "warning", "error"]
    issues: List[ExecutionAuditIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)


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


def _get_state_value(state: Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


def _get_id(record: Dict[str, Any], *keys: str):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _collect_observation_ids(observations: List[Any]) -> set[str]:
    ids = set()

    for obs in observations:
        obs_dict = _as_dict(obs)
        obs_id = _get_id(obs_dict, "observation_id", "id")

        if obs_id:
            ids.add(str(obs_id))

    return ids


def _collect_data_version_ids(data_versions: List[Any]) -> set[str]:
    ids = set()

    for version in data_versions:
        version_dict = _as_dict(version)
        version_id = _get_id(version_dict, "version_id", "id", "data_version_id")

        if version_id:
            ids.add(str(version_id))

    return ids


def _collect_pending_plan_steps(pending_plan: Any) -> Dict[str, Dict[str, Any]]:
    plan = _as_dict(pending_plan)
    steps = {}

    for step in plan.get("steps", []) or []:
        step_dict = _as_dict(step)
        step_id = step_dict.get("step_id")

        if step_id:
            steps[str(step_id)] = step_dict

    return steps


def audit_execution_state(state: Any) -> ExecutionAuditResult:
    """
    Backend-only consistency audit.

    This function does not execute tools, does not call LLMs, and does not mutate state.
    It only checks whether runtime state is internally consistent.
    """
    issues: List[ExecutionAuditIssue] = []

    observations = [
        _as_dict(obs)
        for obs in (_get_state_value(state, "observations", []) or [])
    ]

    analysis_runs = [
        _as_dict(run)
        for run in (_get_state_value(state, "analysis_runs", []) or [])
    ]

    data_versions = [
        _as_dict(version)
        for version in (_get_state_value(state, "data_versions", []) or [])
    ]

    active_data_version_id = _get_state_value(state, "active_data_version_id")
    pending_plan = _get_state_value(state, "pending_plan")
    current_plan_step_id = _get_state_value(state, "current_plan_step_id")
    current_action = _get_state_value(state, "current_action")
    action_origin = _get_state_value(state, "action_origin")

    observation_ids = _collect_observation_ids(observations)
    data_version_ids = _collect_data_version_ids(data_versions)
    pending_steps = _collect_pending_plan_steps(pending_plan)

    # 1. analysis_runs should be structurally valid.
    for idx, run in enumerate(analysis_runs):
        location = f"analysis_runs[{idx}]"

        tool_name = run.get("tool_name")
        status = run.get("status")
        success = run.get("success")
        observation_id = run.get("observation_id")
        data_version_id = run.get("data_version_id")

        if not tool_name:
            issues.append(ExecutionAuditIssue(
                severity="error",
                code="ANALYSIS_RUN_MISSING_TOOL_NAME",
                message="analysis_run is missing tool_name.",
                location=location,
                details={"run": run},
            ))

        if status in {None, ""}:
            issues.append(ExecutionAuditIssue(
                severity="warning",
                code="ANALYSIS_RUN_MISSING_STATUS",
                message="analysis_run is missing status.",
                location=location,
                details={"run": run},
            ))

        if success is True and not data_version_id:
            issues.append(ExecutionAuditIssue(
                severity="warning",
                code="SUCCESSFUL_ANALYSIS_RUN_MISSING_DATA_VERSION",
                message="successful analysis_run is missing data_version_id.",
                location=location,
                details={"run": run},
            ))

        if observation_id and observation_ids and str(observation_id) not in observation_ids:
            issues.append(ExecutionAuditIssue(
                severity="error",
                code="ANALYSIS_RUN_REFERENCES_UNKNOWN_OBSERVATION",
                message="analysis_run references an observation_id that does not exist.",
                location=location,
                details={
                    "observation_id": observation_id,
                    "known_observation_ids": sorted(observation_ids),
                },
            ))

    # 2. active_data_version_id should be valid when data_versions exist.
    if active_data_version_id and data_version_ids:
        if str(active_data_version_id) not in data_version_ids:
            issues.append(ExecutionAuditIssue(
                severity="error",
                code="ACTIVE_DATA_VERSION_NOT_REGISTERED",
                message="active_data_version_id is not present in data_versions.",
                location="active_data_version_id",
                details={
                    "active_data_version_id": active_data_version_id,
                    "known_data_version_ids": sorted(data_version_ids),
                },
            ))

    if active_data_version_id is None and data_versions:
        issues.append(ExecutionAuditIssue(
            severity="warning",
            code="ACTIVE_DATA_VERSION_MISSING",
            message="data_versions exist but active_data_version_id is None.",
            location="active_data_version_id",
        ))

    # 3. pending_plan running step should align with current_plan_step_id.
    running_steps = [
        step_id
        for step_id, step in pending_steps.items()
        if step.get("execution_status") == "running"
    ]

    if running_steps and not current_plan_step_id:
        issues.append(ExecutionAuditIssue(
            severity="warning",
            code="RUNNING_PLAN_STEP_WITHOUT_CURRENT_STEP",
            message="pending_plan contains a running step but current_plan_step_id is missing.",
            location="pending_plan.steps",
            details={"running_steps": running_steps},
        ))

    if current_plan_step_id:
        if pending_steps and str(current_plan_step_id) not in pending_steps:
            issues.append(ExecutionAuditIssue(
                severity="error",
                code="CURRENT_PLAN_STEP_NOT_IN_PENDING_PLAN",
                message="current_plan_step_id does not exist in pending_plan.steps.",
                location="current_plan_step_id",
                details={
                    "current_plan_step_id": current_plan_step_id,
                    "known_step_ids": sorted(pending_steps.keys()),
                },
            ))

        if action_origin == "pending_plan" and current_action is None:
            issues.append(ExecutionAuditIssue(
                severity="warning",
                code="PENDING_PLAN_STEP_WITHOUT_CURRENT_ACTION",
                message="action_origin is pending_plan but current_action is missing.",
                location="current_action",
                details={
                    "current_plan_step_id": current_plan_step_id,
                },
            ))

    if any(issue.severity == "error" for issue in issues):
        status = "error"
    elif issues:
        status = "warning"
    else:
        status = "ok"

    return ExecutionAuditResult(
        status=status,
        issues=issues,
    )