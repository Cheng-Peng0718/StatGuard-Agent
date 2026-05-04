from typing import Any, Dict, List


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