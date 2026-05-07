from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.deliverables.contracts import TaskContract, normalize_task_contract

from core.deliverables.evidence import (
    as_dict,
    get_state_value,
    normalize_string_list,
    extract_final_answer_content_from_state,
    get_satisfied_criterion_names,
    get_satisfied_deliverable_names,
    criterion_satisfied_by_final_answer_text,
)

def _get_contract(state: Any) -> TaskContract:
    contract = get_state_value(state, "task_contract")

    if contract is None:
        contract = get_state_value(state, "deliverable_contract")

    return normalize_task_contract(contract)

class DeliverableGateResult(BaseModel):
    status: Literal["ok", "needs_more_work", "blocked"]
    message: str

    satisfied: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    blocked: List[str] = Field(default_factory=list)

    evidence: Dict[str, Any] = Field(default_factory=dict)

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
        run_dict = as_dict(run)
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
        run_dict = as_dict(run)
        tool_name = run_dict.get("tool_name")

        if tool_name not in required_tools:
            continue

        if not _run_success(run_dict):
            failed.append(tool_name)

    return sorted(set(failed))


def _available_artifact_kinds(analysis_runs: List[Dict[str, Any]]) -> set[str]:
    kinds = set()

    for run in analysis_runs:
        run_dict = as_dict(run)

        for artifact in run_dict.get("artifacts", []) or []:
            artifact_dict = as_dict(artifact)

            kind = (
                artifact_dict.get("artifact_type")
                or artifact_dict.get("type")
                or artifact_dict.get("kind")
            )

            if kind:
                kinds.add(str(kind))

    return kinds

def _get_execution_audit(state: Any) -> Dict[str, Any]:
    return as_dict(get_state_value(state, "execution_audit", {}))


def _execution_audit_codes_by_severity(
    execution_audit: Dict[str, Any],
    severity: str,
) -> List[str]:
    issues = execution_audit.get("issues", []) or []
    codes = []

    for issue in issues:
        issue_dict = as_dict(issue)

        if issue_dict.get("severity") == severity:
            code = issue_dict.get("code")

            if code:
                codes.append(str(code))

    return sorted(set(codes))


def _execution_audit_blockers(execution_audit: Dict[str, Any]) -> List[str]:
    audit_status = execution_audit.get("status")

    if audit_status != "error":
        return []

    error_codes = _execution_audit_codes_by_severity(
        execution_audit,
        severity="error",
    )

    if not error_codes:
        return ["execution_audit:unknown_error"]

    return [
        f"execution_audit:{code}"
        for code in error_codes
    ]


def evaluate_deliverable_gate_state(state: Any) -> DeliverableGateResult:
    """
    Backend evaluator for final-answer deliverable readiness.

    This function does not render UI and does not call tools.
    It only decides whether a final answer can be released.
    """
    contract = _get_contract(state)

    execution_audit = _get_execution_audit(state)
    execution_audit_status = execution_audit.get("status")

    execution_audit_error_codes = _execution_audit_codes_by_severity(
        execution_audit,
        severity="error",
    )
    execution_audit_warning_codes = _execution_audit_codes_by_severity(
        execution_audit,
        severity="warning",
    )
    execution_audit_blockers = _execution_audit_blockers(execution_audit)

    execution_audit_evidence = {
        "execution_audit_status": execution_audit_status,
        "execution_audit_error_codes": execution_audit_error_codes,
        "execution_audit_warning_codes": execution_audit_warning_codes,
    }

    has_contract = any([
        contract.required_tools,
        contract.required_artifacts,
        contract.required_deliverables,
        contract.success_criteria,
    ])

    if not has_contract:
        if execution_audit_blockers:
            return DeliverableGateResult(
                status="needs_more_work",
                message=(
                    "Execution state audit found errors. "
                    "Do not produce a final answer until backend state is consistent."
                ),
                satisfied=[],
                missing=[],
                blocked=execution_audit_blockers,
                evidence={
                    "task_contract_present": False,
                    **execution_audit_evidence,
                },
            )

        return DeliverableGateResult(
            status="ok",
            message="No task_contract declared.",
            satisfied=[],
            missing=[],
            blocked=[],
            evidence={
                "task_contract_present": False,
                **execution_audit_evidence,
            },
        )

    required_tools = contract.required_tools
    required_artifacts = contract.required_artifacts
    required_deliverables = contract.required_deliverables
    success_criteria = contract.success_criteria

    raw_analysis_runs = get_state_value(state, "analysis_runs", []) or []
    analysis_runs = [as_dict(run) for run in raw_analysis_runs]

    successful_tools = _successful_tools(analysis_runs)
    failed_required = _failed_required_tools(analysis_runs, required_tools)
    artifact_kinds = _available_artifact_kinds(analysis_runs)

    final_answer_text = extract_final_answer_content_from_state(state)
    satisfied_deliverable_names = get_satisfied_deliverable_names(state)
    satisfied_criterion_names = get_satisfied_criterion_names(state)

    base_evidence = {
        "task_contract_present": True,
        "required_tools": required_tools,
        "required_artifacts": required_artifacts,
        "required_deliverables": required_deliverables,
        "success_criteria": success_criteria,
        "allow_partial": contract.allow_partial,
        "successful_tools": sorted(successful_tools),
        "artifact_kinds": sorted(artifact_kinds),
        "has_final_answer_text": bool(final_answer_text),
        "satisfied_deliverable_names": sorted(satisfied_deliverable_names),
        "satisfied_criterion_names": sorted(satisfied_criterion_names),
        **execution_audit_evidence,
    }

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

        if deliverable in satisfied_deliverable_names or label in satisfied_deliverable_names:
            satisfied.append(label)
        else:
            missing.append(label)

    # S10B also conservatively marks success criteria missing.
    # S10C will connect these to final-answer content/evidence.
    for criterion in success_criteria:
        label = f"criterion:{criterion}"

        if (
                criterion in satisfied_criterion_names
                or label in satisfied_criterion_names
                or criterion_satisfied_by_final_answer_text(criterion, final_answer_text)
        ):
            satisfied.append(label)
        else:
            missing.append(label)

    for tool_name in failed_required:
        blocked.append(f"tool_failed:{tool_name}")

    for blocker in execution_audit_blockers:
        if blocker not in blocked:
            blocked.append(blocker)

    if blocked:
        return DeliverableGateResult(
            status="needs_more_work",
            message=(
                "Required backend evidence is blocked by failed tools or execution-state errors. "
                "Do not produce a final answer yet."
            ),
            satisfied=satisfied,
            missing=missing,
            blocked=blocked,
            evidence=base_evidence,
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
            evidence=base_evidence,
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
            evidence=base_evidence,
        )

    return DeliverableGateResult(
        status="ok",
        message="All required deliverables are satisfied.",
        satisfied=satisfied,
        missing=[],
        blocked=[],
        evidence=base_evidence,
    )