from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


ExecuteFn = Callable[..., Dict[str, Any]]
ExtractorFn = Callable[..., Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]]
GuardrailFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]
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

    # Column-aware arguments
    column_args: List[str] = field(default_factory=list)
    column_list_args: List[str] = field(default_factory=list)
    allow_all_columns: bool = False

    # Value-domain validation
    allowed_values: Dict[str, List[Any]] = field(default_factory=dict)

    # Conditional validation:
    # Example:
    # {
    #   "action_type": {
    #       "drop": {"strategy": ["rows"]},
    #       "impute": {"strategy": ["mean", "median"]}
    #   }
    # }
    conditional_allowed_values: Dict[str, Dict[Any, Dict[str, List[Any]]]] = field(default_factory=dict)

    # Optional argument aliases / canonicalization.
    # Example:
    # {"row": "rows", "drop_rows": "rows"}
    value_aliases: Dict[str, Dict[Any, Any]] = field(default_factory=dict)

    def canonicalize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize tool arguments before validation/execution.

        This is part of the unified plugin contract.
        It should stay deterministic and non-LLM.
        """
        args = dict(arguments or {})

        for arg_name, alias_map in self.value_aliases.items():
            if arg_name not in args:
                continue

            value = args[arg_name]

            if isinstance(value, str):
                key = value.strip().lower()
            else:
                key = value

            if key in alias_map:
                args[arg_name] = alias_map[key]
            elif isinstance(value, str):
                args[arg_name] = key

        return args

    def to_contract_dict(self) -> Dict[str, Any]:
        """
        New canonical contract representation.

        Use this for validation, prompt tool listing, and tests.
        Avoid adding more legacy-schema adapters.
        """
        return {
            "required": self.required,
            "optional": self.optional,
            "column_args": self.column_args,
            "column_list_args": self.column_list_args,
            "allow_all_columns": self.allow_all_columns,
            "allowed_values": self.allowed_values,
            "conditional_allowed_values": self.conditional_allowed_values,
            "value_aliases": self.value_aliases,
        }

    def to_legacy_schema_dict(self) -> Dict[str, Any]:
        """
        Temporary adapter only.

        Keep this while some old tests still import tools.tool_schema.
        Do not let new runtime code depend on this.
        """
        return self.to_contract_dict()

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


############################################################

@dataclass
class VariableRoleSpec:
    """
    Generic variable-role contract for an analysis method.

    Examples:
    - linear regression: outcome, predictors
    - chi-square: x, y
    - t-test: outcome, group
    - correlation: columns
    """
    role_name: str
    required: bool = True
    user_must_select: bool = True

    allowed_semantic_types: List[str] = field(default_factory=list)

    min_variables: int = 1
    max_variables: Optional[int] = 1

    allow_auto_select: bool = False
    description: str = ""


@dataclass
class ApplicabilityResult:
    """
    Generic result returned by a plugin's applicability checker.

    This is used by planning and capability-map generation,
    not by execution directly.
    """
    status: str
    reason: str

    required_user_choices: List[str] = field(default_factory=list)
    candidate_variables: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "required_user_choices": self.required_user_choices,
            "candidate_variables": self.candidate_variables,
            "warnings": self.warnings,
            "suggested_alternatives": self.suggested_alternatives,
        }


@dataclass
class VersioningPolicy:
    """
    Generic data-versioning contract.

    Mutating plugins must create child data versions.
    Non-mutating analytical plugins must not mutate active data.
    """
    mutates_data: bool = False
    must_create_child_version: bool = False
    allowed_to_call_save_df: bool = False


@dataclass
class RepairPolicy:
    """
    Generic ReAct repair contract.

    This does not mean the agent can modify project source code.
    It only governs analysis-level repair behavior.
    """
    max_attempts: int = 2

    repairable_error_codes: List[str] = field(default_factory=list)
    non_repairable_error_codes: List[str] = field(default_factory=list)

    allow_argument_repair: bool = True
    allow_method_fallback: bool = True
    requires_user_for_missing_roles: bool = True

@dataclass
class PlanningPolicy:
    """
    Generic planning contract.
    """
    include_in_capability_map: bool = True
    ready_without_user_variables: bool = False
    allow_default_arguments: bool = False
    planning_description: str = ""
    requires_variable_contract: bool = True

    # Non-column choices required before a plan step can execute.
    # Examples:
    # - clean_data: action_type, strategy
    # - independent t-test: group1_val, group2_val
    required_user_choices: List[str] = field(default_factory=list)

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
    argument_schema: ArgumentSchema = field(default_factory=ArgumentSchema)

    guardrail_evaluators: List[GuardrailFn] = field(default_factory=list)
    display_config: DisplayConfig = field(default_factory=DisplayConfig)

    # Generic method/planning contract
    method_family: str = "general"
    variable_roles: List[VariableRoleSpec] = field(default_factory=list)
    applicability_checker: Optional[Callable[..., ApplicabilityResult]] = None
    plan_step_builder: Optional[Callable[..., Dict[str, Any]]] = None

    # Generic versioning contract
    mutates_data: bool = False
    versioning_policy: VersioningPolicy = field(default_factory=VersioningPolicy)

    # Generic repair contract
    repair_policy: RepairPolicy = field(default_factory=RepairPolicy)

    # Generic planning policy
    planning_policy: PlanningPolicy = field(default_factory=PlanningPolicy)

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