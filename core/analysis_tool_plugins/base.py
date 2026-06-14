from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


ExecuteFn = Callable[..., Dict[str, Any]]
ExtractorFn = Callable[..., Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]]
GuardrailFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]
# (action, profile, state) -> Optional[(needs_review: bool, reason: str)].
# Returning None defers to the static `requires_confirmation` flag.
ConfirmationFn = Callable[[Any, Any, Any], Optional[Tuple[bool, str]]]
DisplayFormatter = Callable[[Any], Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==========================================================
# Generic formatting helpers
# ==========================================================

def humanize_key(key: Any) -> str:
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


def format_number(value: Any, digits: int = 4) -> Any:
    try:
        v = float(value)
    except Exception:
        return value

    if abs(v) < 1e-10:
        return "0"

    if abs(v) < 0.0001:
        return f"{v:.2e}"

    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def format_p_value(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)

    if v < 0.0001:
        return "<0.0001"

    return f"{v:.4f}".rstrip("0").rstrip(".")


def format_bool_yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def format_list_semicolon(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(x) for x in value)
    return str(value)


# ==========================================================
# Schema config
# ==========================================================

@dataclass
class ArgumentSchema:
    required: Dict[str, type] = field(default_factory=dict)
    optional: Dict[str, type] = field(default_factory=dict)
    column_args: List[str] = field(default_factory=list)
    column_list_args: List[str] = field(default_factory=list)
    allow_all_columns: bool = False

    def to_schema_dict(self) -> Dict[str, Any]:
        """
        Canonical schema dictionary used by analysis_tool_plugins validation.
        """
        return {
            "required": self.required,
            "optional": self.optional,
            "column_args": self.column_args,
            "column_list_args": self.column_list_args,
            "allow_all_columns": self.allow_all_columns,
        }

    def to_legacy_schema_dict(self) -> Dict[str, Any]:
        """
        Backward-compatible alias. Do not use in new code.
        """
        return self.to_schema_dict()

# ==========================================================
# Display config
# ==========================================================

@dataclass
class MetricDisplayConfig:
    labels: Dict[str, str] = field(default_factory=dict)
    formatters: Dict[str, DisplayFormatter] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)


@dataclass
class TableDisplayConfig:
    column_labels: Dict[str, str] = field(default_factory=dict)
    column_formatters: Dict[str, DisplayFormatter] = field(default_factory=dict)
    column_order: List[str] = field(default_factory=list)
    value_mappers: Dict[str, Dict[Any, Any]] = field(default_factory=dict)


@dataclass
class DisplayConfig:
    metrics: MetricDisplayConfig = field(default_factory=MetricDisplayConfig)
    tables: Dict[str, TableDisplayConfig] = field(default_factory=dict)


def metric_rows_from_dict_with_display(
    metrics: Dict[str, Any],
    config: MetricDisplayConfig,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = metrics or {}

    ordered_keys: List[str] = []

    for key in config.order or []:
        if key in metrics and key not in ordered_keys:
            ordered_keys.append(key)

    for key in metrics.keys():
        if key not in ordered_keys:
            ordered_keys.append(key)

    for key in ordered_keys:
        value = metrics.get(key)

        if value is None:
            continue

        label = config.labels.get(key, humanize_key(key))
        formatter = config.formatters.get(key)

        if formatter:
            display_value = formatter(value)
        else:
            display_value = clean_metric_value(value)

        rows.append({
            "label": label,
            "value": display_value,
            "raw_key": key,
        })

    return rows


def normalize_table_from_list_with_display(
    rows: List[Dict[str, Any]],
    config: TableDisplayConfig,
) -> Dict[str, Any]:
    if not rows:
        return {"columns": [], "rows": []}

    raw_columns: List[str] = []

    for key in config.column_order or []:
        if key not in raw_columns:
            raw_columns.append(key)

    for row in rows:
        if isinstance(row, dict):
            for key in row.keys():
                if key not in raw_columns:
                    raw_columns.append(key)

    columns = [
        {
            "label": config.column_labels.get(col, humanize_key(col)),
            "raw_key": col,
        }
        for col in raw_columns
    ]

    normalized_rows = []

    for row in rows:
        normalized_row = []

        for col in raw_columns:
            value = row.get(col, "") if isinstance(row, dict) else ""

            if col in config.value_mappers:
                value = config.value_mappers[col].get(value, value)

            formatter = config.column_formatters.get(col)

            if formatter:
                value = formatter(value)
            else:
                value = clean_metric_value(value)

            normalized_row.append(value)

        normalized_rows.append(normalized_row)

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
    display_config: Optional[DisplayConfig] = None,
) -> List[Dict[str, Any]]:
    display_config = display_config or DisplayConfig()
    blocks: List[Dict[str, Any]] = []

    if summary:
        blocks.append({
            "type": "text",
            "title": "Summary",
            "content": summary,
        })

    metric_rows = metric_rows_from_dict_with_display(
        metrics or {},
        display_config.metrics,
    )

    if metric_rows:
        blocks.append({
            "type": "metric_table",
            "title": "Metrics",
            "rows": metric_rows,
        })

    for table_name, table_data in (tables or {}).items():
        table_display = display_config.tables.get(table_name, TableDisplayConfig())

        if isinstance(table_data, list) and all(isinstance(x, dict) for x in table_data):
            normalized = normalize_table_from_list_with_display(
                table_data,
                table_display,
            )

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


# ==========================================================
# Default extraction
# ==========================================================

def default_extractor(
    *,
    tool_name: str,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = default_title
    summary = default_summary

    metrics: Dict[str, Any] = {}
    tables: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    if isinstance(payload, dict) and payload:
        tables["payload"] = payload

    return title, summary, metrics, tables, metadata


# ==========================================================
# Unified plugin
# ==========================================================

@dataclass
class AnalysisToolPlugin:
    tool_name: str
    display_name: str

    execute: Optional[ExecuteFn] = None
    extractor: Optional[ExtractorFn] = None

    requires_confirmation: bool = False
    is_inferential: bool = False
    argument_schema: ArgumentSchema = field(default_factory=ArgumentSchema)

    guardrail_evaluators: List[GuardrailFn] = field(default_factory=list)
    # Optional per-action review gate. When set, it decides needs_review/allowed
    # (overriding requires_confirmation); returning None defers to the flag.
    confirmation_policy: Optional[ConfirmationFn] = None
    display_config: DisplayConfig = field(default_factory=DisplayConfig)

    description: str = ""
    usage_guidance: str = ""
    use_when: List[str] = field(default_factory=list)
    do_not_use_when: List[str] = field(default_factory=list)

    # "any" | "dataframe" | "sql"
    requires_data_source: str = "any"

    # True if this tool can create/update the active workspace dataset.
    produces_active_dataset: bool = False

    # Lightweight examples shown to the Supervisor.
    examples: List[Dict[str, Any]] = field(default_factory=list)

    # High-level evidence categories produced by this tool.
    # These are used by the answer quality gate and coverage brief.
    evidence_categories: List[str] = field(default_factory=list)

    # Role of each evidence category produced by this tool.
    # Allowed roles:
    # - substantive: final answer-supporting analysis evidence
    # - pre_analysis_check: readiness/data-quality check; warning source, not hard final coverage
    # - provenance: data source / materialization evidence; report provenance, not hard final coverage
    # - optional_context: useful context, not hard final coverage
    evidence_category_roles: Dict[str, str] = field(default_factory=dict)

    # Optional: function that produces an APA-style Methods/Results paragraph
    # from an analysis_run dict. Signature: (run: Dict) -> Optional[str].
    # If None, the export_apa_methods plugin will skip this tool.
    apa_methods_writer: Optional[Any] = None

    def run(self, context) -> Dict[str, Any]:
        if self.execute is None:
            return {
                "status": "failed",
                "error_code": "MISSING_PLUGIN_EXECUTE",
                "message": f"Plugin `{self.tool_name}` does not define an execute function.",
                "recoverable": False,
                "details": {},
                "artifacts": [],
            }

        return self.execute(context)

    def extract(
        self,
        *,
        payload: Dict[str, Any],
        arguments: Dict[str, Any],
        default_title: str,
        default_summary: str,
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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

        if status in {"blocked", "failed"}:
            title = default_title

            if status == "blocked":
                title = f"{default_title} (Blocked)"
            elif status == "failed":
                title = f"{default_title} (Failed)"

            summary = default_summary
            metrics = {
                "status": status,
                "success": success,
            }

            tables = {}
            if payload:
                tables["details"] = payload

            metadata = {
                "blocked_or_failed": True,
            }
        else:
            title, summary, metrics, tables, metadata = self.extract(
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
            display_config=self.display_config,
        )

        analysis_run = {
            "run_id": f"run_{uuid.uuid4().hex[:8]}",
            "tool_name": self.tool_name,
            "evidence_categories": list(self.evidence_categories or []),
            "evidence_category_roles": dict(self.evidence_category_roles or {}),
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

        analysis_run["guardrails"] = self.evaluate_guardrails(analysis_run)

        return analysis_run