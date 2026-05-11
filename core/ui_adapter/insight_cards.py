from __future__ import annotations

from typing import Any, Dict, List

from core.ui_adapter.insight_specs import get_insight_spec


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _format_metric_findings(
    metrics: Dict[str, Any],
    metric_labels: Dict[str, str],
    *,
    max_items: int = 5,
) -> List[str]:
    findings = []

    for key, value in (metrics or {}).items():
        if value is None:
            continue

        if not isinstance(value, (int, float, str, bool)):
            continue

        label = metric_labels.get(key, key.replace("_", " "))

        findings.append(f"{label}: {value}")

        if len(findings) >= max_items:
            break

    return findings


def _format_guardrail_findings(guardrails: List[Dict[str, Any]]) -> List[str]:
    findings = []

    for item in guardrails or []:
        title = item.get("title")
        message = item.get("message")
        severity = item.get("severity")

        if title and message:
            text = f"{title}: {message}"
        elif title:
            text = str(title)
        elif message:
            text = str(message)
        else:
            continue

        if severity:
            text = f"[{severity}] {text}"

        findings.append(text)

    return findings


def build_insight_card_from_run(run: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(run, dict):
        raise TypeError("build_insight_card_from_run requires a run dictionary.")

    tool_name = run.get("tool_name")
    spec = get_insight_spec(tool_name)

    success = run.get("success")
    status = run.get("status")
    error_code = run.get("error_code")
    summary = run.get("summary") or run.get("message")

    metrics = run.get("metrics") or {}
    guardrails = _as_list(run.get("guardrails"))

    card = {
        "tool_name": tool_name,
        "title": spec.display_name,
        "status": status,
        "success": success,
        "what_was_computed": spec.what_was_computed,
        "key_findings": [],
        "caveats": [],
        "recommended_next_steps": [],
        "metadata": {
            "analysis_run_id": run.get("analysis_run_id"),
            "observation_id": run.get("observation_id"),
            "data_version_id": run.get("data_version_id"),
            "error_code": error_code,
        },
    }

    if success is False:
        card["key_findings"].append(
            summary or f"{spec.display_name} did not complete successfully."
        )

        if error_code:
            card["key_findings"].append(f"Error code: {error_code}")

        card["caveats"].append(
            "This failed run should not be interpreted as a valid statistical result."
        )

        card["recommended_next_steps"].append(
            "Check the selected variables, sample size, missing values, and tool requirements before rerunning."
        )

        return card

    if summary:
        card["key_findings"].append(str(summary))

    card["key_findings"].extend(
        _format_metric_findings(
            metrics,
            spec.metric_labels,
        )
    )

    card["caveats"].extend(
        _format_guardrail_findings(guardrails)
    )

    if not card["caveats"]:
        card["caveats"].extend(spec.default_caveats)

    card["recommended_next_steps"].extend(
        spec.recommended_next_steps
    )

    if not card["key_findings"]:
        card["key_findings"].append(
            "The run completed, but no detailed findings were available."
        )

    return card


def build_latest_insight_card_from_state(state: Dict[str, Any]) -> Dict[str, Any] | None:
    runs = state.get("analysis_runs") or []

    if not runs:
        return None

    return build_insight_card_from_run(runs[-1])


def build_insight_cards_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        build_insight_card_from_run(run)
        for run in (state.get("analysis_runs") or [])
    ]