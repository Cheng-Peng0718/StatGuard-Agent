from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


ExtractorFn = Callable[..., Tuple[str, str, Dict[str, Any], Dict[str, Any]]]
GuardrailFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def humanize_key(key: Any) -> str:
    """
    Generic key humanizer.

    This must not contain dataset-specific or analysis-specific mappings.
    """
    if key is None:
        return ""

    text = str(key).strip()
    if not text:
        return ""

    if text.isupper():
        return text

    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())

    return text[:1].upper() + text[1:]


def compact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if v is not None}


def safe_join_list(values: Any, sep: str = " + ") -> str:
    if isinstance(values, list):
        return sep.join(str(x) for x in values)
    if isinstance(values, str):
        return values
    return ""


def clean_metric_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return value


def metric_rows_from_dict(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    for key, value in (metrics or {}).items():
        if value is None:
            continue

        rows.append({
            "label": humanize_key(key),
            "value": clean_metric_value(value),
            "raw_key": key,
        })

    return rows


def normalize_table_from_list(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"columns": [], "rows": []}

    raw_columns = []

    for row in rows:
        if isinstance(row, dict):
            for key in row.keys():
                if key not in raw_columns:
                    raw_columns.append(key)

    columns = [
        {
            "label": humanize_key(col),
            "raw_key": col,
        }
        for col in raw_columns
    ]

    normalized_rows = []

    for row in rows:
        normalized_rows.append([
            clean_metric_value(row.get(col, ""))
            for col in raw_columns
        ])

    return {
        "columns": columns,
        "rows": normalized_rows,
    }


def build_generic_report_blocks(
    *,
    summary: str,
    metrics: Dict[str, Any],
    tables: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert extracted metrics/tables/artifacts into generic report blocks.

    This function is method-agnostic.
    """
    blocks: List[Dict[str, Any]] = []

    if summary:
        blocks.append({
            "type": "text",
            "title": "Summary",
            "content": summary,
        })

    metric_rows = metric_rows_from_dict(metrics or {})
    if metric_rows:
        blocks.append({
            "type": "metric_table",
            "title": "Metrics",
            "rows": metric_rows,
        })

    for table_name, table_data in (tables or {}).items():
        if isinstance(table_data, list) and all(isinstance(x, dict) for x in table_data):
            normalized = normalize_table_from_list(table_data)
            blocks.append({
                "type": "table",
                "title": humanize_key(table_name),
                "columns": normalized["columns"],
                "rows": normalized["rows"],
            })
        elif isinstance(table_data, dict):
            blocks.append({
                "type": "json",
                "title": humanize_key(table_name),
                "content": table_data,
            })
        else:
            blocks.append({
                "type": "text",
                "title": humanize_key(table_name),
                "content": str(table_data),
            })

    for artifact in artifacts or []:
        artifact_type = artifact.get("type")

        if artifact_type in {"png", "jpg", "jpeg"}:
            blocks.append({
                "type": "figure",
                "title": artifact.get("name") or "Figure",
                "path": artifact.get("path"),
                "name": artifact.get("name"),
                "artifact_type": artifact_type,
            })
        else:
            blocks.append({
                "type": "artifact",
                "title": artifact.get("name") or "Artifact",
                "content": artifact,
            })

    return blocks


def default_extractor(
    *,
    tool_name: str,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """
    Fallback extractor for unregistered tools.
    """
    title = default_title
    summary = default_summary

    metrics: Dict[str, Any] = {}
    tables: Dict[str, Any] = {}

    if isinstance(payload, dict) and payload:
        tables["payload"] = payload

    return title, summary, metrics, tables


@dataclass
class AnalysisPlugin:
    tool_name: str
    display_name: str
    extractor: Optional[ExtractorFn] = None
    guardrail_evaluators: List[GuardrailFn] = field(default_factory=list)

    def extract(
        self,
        *,
        payload: Dict[str, Any],
        arguments: Dict[str, Any],
        default_title: str,
        default_summary: str,
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        if self.extractor is None:
            return default_extractor(
                tool_name=self.tool_name,
                payload=payload,
                arguments=arguments,
                default_title=default_title,
                default_summary=default_summary,
            )

        return self.extractor(
            payload=payload,
            arguments=arguments,
            default_title=default_title,
            default_summary=default_summary,
        )

    def evaluate_guardrails(self, analysis_run: Dict[str, Any]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []

        for evaluator in self.guardrail_evaluators:
            try:
                findings.extend(evaluator(analysis_run) or [])
            except Exception as e:
                findings.append({
                    "finding_id": f"gr_{uuid.uuid4().hex[:8]}",
                    "category": "guardrail_execution",
                    "severity": "warning",
                    "title": "Guardrail evaluator failed",
                    "message": f"A guardrail evaluator failed for tool `{self.tool_name}`.",
                    "evidence": {
                        "error": str(e),
                        "evaluator": getattr(evaluator, "__name__", str(evaluator)),
                    },
                    "recommendation": "Inspect the guardrail evaluator implementation.",
                })

        return findings

    def build_analysis_run(
        self,
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

        default_title = self.display_name or self.tool_name.replace("_", " ").title()
        default_summary = f"Tool `{self.tool_name}` finished with status `{status}`."
        if message:
            default_summary += f" {message}"

        title, summary, metrics, tables = self.extract(
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
        )

        analysis_run = {
            "run_id": f"run_{uuid.uuid4().hex[:8]}",
            "tool_name": self.tool_name,
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
            "report_blocks": report_blocks,
            "guardrails": [],
            "raw_observation_id": observation_id,
        }

        analysis_run["guardrails"] = self.evaluate_guardrails(analysis_run)

        return analysis_run