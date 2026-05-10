from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.analysis_tool_plugins.reporting import (
    build_generic_report_blocks,
    default_extractor,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def evaluate_guardrails_for_plugin(
    plugin: Any,
    analysis_run: Dict[str, Any],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for evaluator in getattr(plugin, "guardrail_evaluators", []) or []:
        try:
            findings.extend(evaluator(analysis_run) or [])
        except Exception as exc:
            findings.append({
                "finding_id": f"gr_{uuid.uuid4().hex[:8]}",
                "category": "guardrail_execution",
                "severity": "warning",
                "title": "Guardrail evaluator failed",
                "message": (
                    f"A guardrail evaluator failed for tool `{plugin.tool_name}`."
                ),
                "evidence": {
                    "error": str(exc),
                    "evaluator": getattr(evaluator, "__name__", str(evaluator)),
                },
                "recommendation": (
                    "Inspect the guardrail evaluator implementation."
                ),
            })

    return findings

def build_analysis_run_for_plugin(
    plugin: Any,
    *,
    action_id: str,
    arguments: Dict[str, Any],
    data_version_id: Optional[str],
    status: str,
    success: bool,
    message: Optional[str],
    payload: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    observation_id: str,
) -> Dict[str, Any]:
    arguments = arguments or {}
    payload = payload or {}
    artifacts = artifacts or []

    default_title = plugin.display_name or plugin.tool_name.replace("_", " ").title()
    default_summary = f"Tool `{plugin.tool_name}` finished with status `{status}`."
    if message:
        default_summary += f" {message}"

    extractor = getattr(plugin, "extractor", None)

    if extractor is None:
        title, summary, metrics, tables, metadata = default_extractor(
            tool_name=plugin.tool_name,
            payload=payload,
            arguments=arguments,
            default_title=default_title,
            default_summary=default_summary,
        )
    else:
        title, summary, metrics, tables, metadata = extractor(
            payload=payload,
            arguments=arguments,
            default_title=default_title,
            default_summary=default_summary,
        )

    report_blocks = build_generic_report_blocks(
        summary=summary,
        metrics=metrics,
        tables=tables,
        artifacts=artifacts,
        display_config=getattr(plugin, "display_config", None),
    )

    analysis_run = {
        "run_id": f"run_{uuid.uuid4().hex[:8]}",
        "tool_name": plugin.tool_name,
        "action_id": action_id,
        "data_version_id": data_version_id,
        "status": status,
        "success": success,
        "created_at": utc_now_iso(),
        "title": title,
        "summary": summary,
        "arguments": arguments,
        "metrics": metrics,
        "tables": tables,
        "artifacts": artifacts,
        "metadata": metadata,
        "report_blocks": report_blocks,
        "guardrails": [],
        "raw_observation_id": observation_id,
    }

    analysis_run["guardrails"] = evaluate_guardrails_for_plugin(plugin, analysis_run)
    return analysis_run
