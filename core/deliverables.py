from typing import Any, Dict, List
from core.analysis_coverage import (
    available_evidence_categories_from_plugins,
    covered_evidence_categories_from_runs,
    missing_required_evidence_categories,
    normalize_coverage_brief,
)

def _get_obs_payload(obs: Dict[str, Any]) -> Dict[str, Any]:
    structured = obs.get("structured_data", {}) or {}
    payload = structured.get("payload", {}) or {}

    # 新工具通常 payload 本身含 details
    if isinstance(payload, dict):
        return payload

    return {}


def _get_obs_details(obs: Dict[str, Any]) -> Dict[str, Any]:
    payload = _get_obs_payload(obs)

    if isinstance(payload, dict):
        if "details" in payload and isinstance(payload["details"], dict):
            return payload["details"]
        return payload

    return {}


def _obs_status_ok(obs: Dict[str, Any]) -> bool:
    status = obs.get("status")
    if status is None:
        status = (obs.get("structured_data", {}) or {}).get("status")
    return status in {"ok", "warning"}


def _obs_artifacts(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts = obs.get("artifacts")
    if artifacts is None:
        artifacts = (obs.get("structured_data", {}) or {}).get("artifacts", [])
    return artifacts or []


def _evidence_satisfied(evidence: str, obs: Dict[str, Any]) -> bool:
    details = _get_obs_details(obs)
    artifacts = _obs_artifacts(obs)

    if evidence == "status_ok":
        return _obs_status_ok(obs)

    if evidence == "png_artifact":
        return any(a.get("type") == "png" and a.get("path") for a in artifacts)

    if evidence == "coef_table":
        return "coef_table" in details or "coefficients" in details

    if evidence == "r_squared":
        return "r_squared" in details or "R_squared" in details

    if evidence == "vif":
        return "vif" in details or "vif_table" in details

    if evidence == "breusch_pagan":
        return "breusch_pagan" in details or "breusch_pagan_test" in details

    if evidence == "residual_summary":
        keys = {
            "residual_mean",
            "residual_std",
            "residual_skewness",
            "residual_kurtosis_fisher",
            "diagnostic_flags",
        }
        return any(k in details for k in keys)

    # Generic fallback: check if key exists in details.
    return evidence in details


def check_deliverables(task_contract: Dict[str, Any], observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generic completion checker.
    It does not parse user_request.
    It only checks the LLM-declared task_contract against observations.
    """
    if not task_contract:
        return {
            "status": "ok",
            "message": "No task_contract declared.",
            "satisfied": [],
            "missing": [],
            "blocked": [],
        }

    required = task_contract.get("required_deliverables", []) or []
    satisfied = []
    missing = []
    blocked = []

    for deliverable in required:
        deliverable_id = deliverable.get("deliverable_id") or deliverable.get("id")
        satisfied_by = deliverable.get("satisfied_by", []) or []
        required_evidence = deliverable.get("required_evidence", []) or ["status_ok"]

        deliverable_status = deliverable.get("status", "pending")

        if deliverable_status == "blocked":
            blocked.append({
                "deliverable_id": deliverable_id,
                "description": deliverable.get("description"),
                "reason": "deliverable_marked_blocked_by_contract",
                "satisfied_by": satisfied_by,
                "required_evidence": required_evidence,
            })
            continue

        candidate_obs = [
            obs for obs in observations
            if obs.get("tool_name") in satisfied_by
        ]

        successful_candidates = [
            obs for obs in candidate_obs
            if _obs_status_ok(obs)
        ]

        if not successful_candidates:
            missing.append({
                "deliverable_id": deliverable_id,
                "description": deliverable.get("description"),
                "reason": "no_successful_observation_from_required_tools",
                "satisfied_by": satisfied_by,
                "required_evidence": required_evidence,
            })
            continue

        # A deliverable is satisfied if any successful candidate has all required evidence.
        matched = None
        missing_evidence_for_best = []

        for obs in successful_candidates:
            missing_evidence = [
                ev for ev in required_evidence
                if not _evidence_satisfied(ev, obs)
            ]

            if not missing_evidence:
                matched = obs
                break

            if not missing_evidence_for_best or len(missing_evidence) < len(missing_evidence_for_best):
                missing_evidence_for_best = missing_evidence

        if matched is None:
            missing.append({
                "deliverable_id": deliverable_id,
                "description": deliverable.get("description"),
                "reason": "missing_required_evidence",
                "satisfied_by": satisfied_by,
                "required_evidence": required_evidence,
                "missing_evidence": missing_evidence_for_best,
            })
        else:
            satisfied.append({
                "deliverable_id": deliverable_id,
                "description": deliverable.get("description"),
                "tool_name": matched.get("tool_name"),
                "observation_id": matched.get("observation_id"),
            })

    if blocked:
        status = "blocked"
    elif missing:
        status = "missing"
    else:
        status = "ok"

    message = (
        "All deliverables satisfied."
        if status == "ok"
        else "Some deliverables are blocked."
        if status == "blocked"
        else "Some deliverables are missing."
    )

    return {
        "status": status,
        "message": message,
        "satisfied": satisfied,
        "missing": missing,
        "blocked": blocked,
    }

DATA_PREP_TOOL_NAMES = {
    "inspect_sql_schema",
    "materialize_sql_query_result",
    "inspect_dataset",
    "clean_data",
}


def _get_action_type(action: Any) -> str | None:
    if action is None:
        return None

    if isinstance(action, dict):
        return action.get("action_type")

    return getattr(action, "action_type", None)


def _run_status(run: Dict[str, Any]) -> str | None:
    return run.get("status")


def _run_tool_name(run: Dict[str, Any]) -> str | None:
    return run.get("tool_name")


def _run_data_version_id(run: Dict[str, Any]) -> str | None:
    return (
        run.get("data_version_id")
        or run.get("input_data_version_id")
        or run.get("produced_data_version_id")
    )


def _is_recorded_analysis_run(run: Dict[str, Any]) -> bool:
    return _run_status(run) in {"ok", "warning", "blocked"}


def _is_substantive_analysis_run(run: Dict[str, Any]) -> bool:
    tool_name = _run_tool_name(run)

    if tool_name in DATA_PREP_TOOL_NAMES:
        return False

    return _is_recorded_analysis_run(run)


def _has_summary(run: Dict[str, Any]) -> bool:
    summary = run.get("summary")
    return isinstance(summary, str) and bool(summary.strip())


def _has_assumption_or_limitation_evidence(run: Dict[str, Any]) -> bool:
    tables = run.get("tables", {}) or {}
    metadata = run.get("metadata", {}) or {}
    guardrails = run.get("guardrails", []) or []

    table_keys = {str(key).lower() for key in tables.keys()}

    if any("assumption" in key or "limitation" in key for key in table_keys):
        return True

    if metadata.get("assumptions_and_limitations"):
        return True

    if guardrails:
        return True

    return False


def _looks_statistical_run(run: Dict[str, Any]) -> bool:
    metrics = run.get("metrics", {}) or {}
    tables = run.get("tables", {}) or {}

    metric_keys = {str(key).lower() for key in metrics.keys()}
    table_keys = {str(key).lower() for key in tables.keys()}

    statistical_metric_signals = {
        "p_value",
        "f_p_value",
        "r_squared",
        "adj_r_squared",
        "effect_size",
        "f_statistic",
        "t_statistic",
        "model_significant_at_alpha",
    }

    if metric_keys.intersection(statistical_metric_signals):
        return True

    if any("coef" in key or "coefficient" in key for key in table_keys):
        return True

    return False


def _quality_check(check_id: str, status: str, message: str, recommendation: str | None = None) -> Dict[str, Any]:
    item = {
        "check_id": check_id,
        "status": status,
        "message": message,
    }

    if recommendation:
        item["recommendation"] = recommendation

    return item


def check_answer_quality(
    *,
    user_request: str,
    current_action: Any,
    analysis_runs: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
    active_data_version_id: str | None = None,
    analysis_coverage_brief: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Soft final-answer quality gate for the workbench-style analyst loop.

    This does not depend on task_contract or pending plan steps.
    It checks whether the final answer is grounded in recorded analysis evidence.
    The first version is intentionally soft: it records warnings but does not
    force the graph back into a workflow loop.
    """
    action_type = _get_action_type(current_action)

    checks: List[Dict[str, Any]] = []

    if action_type == "ask_user":
        checks.append(_quality_check(
            "ask_user_allowed",
            "pass",
            "The assistant is asking the user for clarification or missing information.",
        ))

        return {
            "status": "ok",
            "gate_type": "answer_quality_gate",
            "quality_status": "pass",
            "message": "Answer quality gate passed for an ask_user action.",
            "checks": checks,
            "continuation_recommended": False,
            "coverage_brief": {},
            "available_evidence_categories": available_evidence_categories_from_plugins(),
            "covered_evidence_categories": [],
            "missing_evidence_categories": [],
            "warnings": [],
            "satisfied": [],
            "missing": [],
            "blocked": [],
        }

    recorded_runs = [
        run for run in analysis_runs or []
        if _is_recorded_analysis_run(run)
    ]

    substantive_runs = [
        run for run in recorded_runs
        if _is_substantive_analysis_run(run)
    ]

    blocked_runs = [
        run for run in recorded_runs
        if _run_status(run) == "blocked"
    ]

    warnings: List[Dict[str, Any]] = []

    available_evidence_categories = available_evidence_categories_from_plugins()

    coverage_brief = normalize_coverage_brief(
        analysis_coverage_brief,
        allowed_categories=available_evidence_categories,
        drop_unknown=True,
    )

    covered_evidence_categories = covered_evidence_categories_from_runs(
        analysis_runs or []
    )

    missing_evidence_categories = missing_required_evidence_categories(
        coverage_brief=coverage_brief,
        analysis_runs=analysis_runs or [],
        allowed_categories=available_evidence_categories,
    )

    if recorded_runs:
        checks.append(_quality_check(
            "analysis_runs_recorded",
            "pass",
            f"{len(recorded_runs)} recorded analysis run(s) are available for the final answer.",
        ))
    else:
        warning = _quality_check(
            "analysis_runs_recorded",
            "warn",
            "No recorded analysis runs are available for the final answer.",
            "Final answers for data analysis requests should be grounded in tool observations or analysis runs.",
        )
        checks.append(warning)
        warnings.append(warning)

    if substantive_runs:
        checks.append(_quality_check(
            "substantive_analysis_present",
            "pass",
            f"{len(substantive_runs)} substantive analysis run(s) are available.",
        ))
    else:
        warning = _quality_check(
            "substantive_analysis_present",
            "warn",
            "Only data-preparation or schema-inspection runs were found; no substantive analysis result was recorded.",
            "If the user asked for analysis, run an appropriate analysis tool before finalizing.",
        )
        checks.append(warning)
        warnings.append(warning)

    if active_data_version_id and substantive_runs:
        matching_version_runs = [
            run for run in substantive_runs
            if _run_data_version_id(run) in {active_data_version_id, None, "N/A"}
        ]

        if matching_version_runs:
            checks.append(_quality_check(
                "active_data_version_alignment",
                "pass",
                f"At least one substantive analysis run is aligned with the active data version `{active_data_version_id}`.",
            ))
        else:
            warning = _quality_check(
                "active_data_version_alignment",
                "warn",
                f"No substantive analysis run clearly matches the active data version `{active_data_version_id}`.",
                "Recompute stale results or make the data-version limitation explicit.",
            )
            checks.append(warning)
            warnings.append(warning)

    for run in substantive_runs:
        if not _has_summary(run):
            warning = _quality_check(
                "analysis_summary_present",
                "warn",
                f"Analysis run `{run.get('title') or run.get('tool_name')}` has no human-readable summary.",
                "Each substantive analysis run should provide a concise summary for the final answer and report.",
            )
            checks.append(warning)
            warnings.append(warning)

    statistical_runs = [
        run for run in substantive_runs
        if _looks_statistical_run(run)
    ]

    for run in statistical_runs:
        if not _has_assumption_or_limitation_evidence(run):
            warning = _quality_check(
                "statistical_limitations_present",
                "warn",
                f"Statistical run `{run.get('title') or run.get('tool_name')}` does not expose assumptions, limitations, or guardrails.",
                "Statistical conclusions should include assumptions, limitations, or guardrail findings.",
            )
            checks.append(warning)
            warnings.append(warning)

    if blocked_runs:
        warning = _quality_check(
            "blocked_attempts_visible",
            "warn",
            f"{len(blocked_runs)} blocked analysis attempt(s) are recorded.",
            "The final answer should explain why the requested analysis was blocked and suggest a next step.",
        )
        checks.append(warning)
        warnings.append(warning)

    if coverage_brief and missing_evidence_categories:
        warning = _quality_check(
            "requested_evidence_coverage",
            "warn",
            "The final answer does not yet cover all evidence categories required by the analysis coverage brief.",
            (
                    "Continue analysis by calling an appropriate tool whose plugin-declared "
                    "evidence_categories cover: "
                    + ", ".join(missing_evidence_categories)
            ),
        )
        checks.append(warning)
        warnings.append(warning)

    quality_status = "needs_attention" if warnings else "pass"

    return {
        "status": "ok",
        "gate_type": "answer_quality_gate",
        "quality_status": quality_status,
        "continuation_recommended": bool(missing_evidence_categories),
        "coverage_brief": coverage_brief,
        "available_evidence_categories": available_evidence_categories,
        "covered_evidence_categories": covered_evidence_categories,
        "missing_evidence_categories": missing_evidence_categories,
        "message": (
            "Answer quality gate completed with warnings."
            if warnings
            else "Answer quality gate passed."
        ),
        "checks": checks,
        "warnings": warnings,
        "satisfied": [],
        "missing": [],
        "blocked": [],
    }