from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.deliverables.contracts import TaskContract, normalize_task_contract

class DeliverableGateResult(BaseModel):
    status: Literal["ok", "needs_more_work", "blocked"]
    message: str

    satisfied: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    blocked: List[str] = Field(default_factory=list)

    evidence: Dict[str, Any] = Field(default_factory=dict)


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


def _get_contract(state: Any) -> TaskContract:
    contract = _get_state_value(state, "task_contract")

    if contract is None:
        contract = _get_state_value(state, "deliverable_contract")

    return normalize_task_contract(contract)


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [str(v) for v in value if v is not None]

    if isinstance(value, tuple):
        return [str(v) for v in value if v is not None]

    return []


def _run_success(run: Dict[str, Any]) -> bool:
    status = run.get("status")
    success = run.get("success")

    if success is False:
        return False

    if status in {"failed", "error", "rejected"}:
        return False

    return status in {"ok", "warning", "success", "completed"} or success is True


def _successful_tools(analysis_runs: List[Dict[str, Any]]) -> set[str]:
    tools = set()

    for run in analysis_runs:
        run_dict = _as_dict(run)
        tool_name = run_dict.get("tool_name")

        if tool_name and _run_success(run_dict):
            tools.add(tool_name)

    return tools


def _failed_required_tools(
    analysis_runs: List[Dict[str, Any]],
    required_tools: List[str],
) -> List[str]:
    failed = []

    for run in analysis_runs:
        run_dict = _as_dict(run)
        tool_name = run_dict.get("tool_name")

        if tool_name not in required_tools:
            continue

        if not _run_success(run_dict):
            failed.append(tool_name)

    return sorted(set(failed))


def _available_artifact_kinds(analysis_runs: List[Dict[str, Any]]) -> set[str]:
    kinds = set()

    for run in analysis_runs:
        run_dict = _as_dict(run)

        for artifact in run_dict.get("artifacts", []) or []:
            artifact_dict = _as_dict(artifact)

            kind = (
                artifact_dict.get("artifact_type")
                or artifact_dict.get("type")
                or artifact_dict.get("kind")
            )

            if kind:
                kinds.add(str(kind))

    return kinds


def evaluate_deliverable_gate_state(state: Any) -> DeliverableGateResult:
    """
    Backend evaluator for final-answer deliverable readiness.

    This function does not render UI and does not call tools.
    It only decides whether a final answer can be released.
    """
    contract = _get_contract(state)

    has_contract = any([
        contract.required_tools,
        contract.required_artifacts,
        contract.required_deliverables,
        contract.success_criteria,
    ])

    if not has_contract:
        return DeliverableGateResult(
            status="ok",
            message="No task_contract declared.",
            satisfied=[],
            missing=[],
            blocked=[],
            evidence={
                "task_contract_present": False,
            },
        )

    required_tools = contract.required_tools
    required_artifacts = contract.required_artifacts
    required_deliverables = contract.required_deliverables
    success_criteria = contract.success_criteria

    raw_analysis_runs = _get_state_value(state, "analysis_runs", []) or []
    analysis_runs = [_as_dict(run) for run in raw_analysis_runs]

    successful_tools = _successful_tools(analysis_runs)
    failed_required = _failed_required_tools(analysis_runs, required_tools)
    artifact_kinds = _available_artifact_kinds(analysis_runs)

    satisfied = []
    missing = []
    blocked = []

    for tool_name in required_tools:
        label = f"tool:{tool_name}"

        if tool_name in successful_tools:
            satisfied.append(label)
        else:
            missing.append(label)

    for artifact_kind in required_artifacts:
        label = f"artifact:{artifact_kind}"

        if artifact_kind in artifact_kinds:
            satisfied.append(label)
        else:
            missing.append(label)

    # Contract-level deliverables cannot be considered satisfied
    # merely because tools ran. They need explicit evidence later.
    # S10B conservatively marks them missing.
    for deliverable in required_deliverables:
        label = f"deliverable:{deliverable}"

        if label not in missing:
            missing.append(label)

    # S10B also conservatively marks success criteria missing.
    # S10C will connect these to final-answer content/evidence.
    for criterion in success_criteria:
        label = f"criterion:{criterion}"

        if label not in missing:
            missing.append(label)

    for tool_name in failed_required:
        blocked.append(f"tool_failed:{tool_name}")

    if blocked:
        return DeliverableGateResult(
            status="needs_more_work",
            message=(
                "Some required analysis tools failed or produced unusable results. "
                "Do not produce a final answer yet."
            ),
            satisfied=satisfied,
            missing=missing,
            blocked=blocked,
            evidence={
                "task_contract_present": True,
                "required_tools": required_tools,
                "required_artifacts": required_artifacts,
                "required_deliverables": required_deliverables,
                "successful_tools": sorted(successful_tools),
                "artifact_kinds": sorted(artifact_kinds),
                "success_criteria": success_criteria,
                "allow_partial": contract.allow_partial,
            },
        )

    if missing and contract.allow_partial and not blocked:
        return DeliverableGateResult(
            status="ok",
            message=(
                "Some deliverables are missing, but task_contract.allow_partial=True. "
                "Final answer may proceed with limitations clearly stated."
            ),
            satisfied=satisfied,
            missing=missing,
            blocked=[],
            evidence={
                "task_contract_present": True,
                "required_tools": required_tools,
                "required_artifacts": required_artifacts,
                "required_deliverables": required_deliverables,
                "success_criteria": success_criteria,
                "allow_partial": contract.allow_partial,
                "successful_tools": sorted(successful_tools),
                "artifact_kinds": sorted(artifact_kinds),
            },
        )

    if missing:
        return DeliverableGateResult(
            status="needs_more_work",
            message=(
                "Required deliverables are missing. "
                "Do not produce a final answer yet."
            ),
            satisfied=satisfied,
            missing=missing,
            blocked=[],
            evidence={
                "task_contract_present": True,
                "required_tools": required_tools,
                "required_artifacts": required_artifacts,
                "required_deliverables": required_deliverables,
                "successful_tools": sorted(successful_tools),
                "artifact_kinds": sorted(artifact_kinds),
            },
        )

    return DeliverableGateResult(
        status="ok",
        message="All required deliverables are satisfied.",
        satisfied=satisfied,
        missing=[],
        blocked=[],
        evidence={
            "task_contract_present": True,
            "required_tools": required_tools,
            "required_artifacts": required_artifacts,
            "required_deliverables": required_deliverables,
            "successful_tools": sorted(successful_tools),
            "artifact_kinds": sorted(artifact_kinds),
        },
    )