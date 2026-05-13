from __future__ import annotations

import re
from typing import Any, Dict, List

from core.analysis_tool_plugins.registry import (
    get_available_evidence_categories,
    get_available_evidence_category_roles,
)

def available_evidence_category_roles_from_plugins() -> Dict[str, str]:
    raw_roles = get_available_evidence_category_roles()
    roles: Dict[str, str] = {}

    for category, role in raw_roles.items():
        normalized_category = normalize_evidence_category(category)
        normalized_role = str(role or "substantive").strip().lower()

        if not normalized_category:
            continue

        if normalized_role not in {
            "substantive",
            "pre_analysis_check",
            "provenance",
            "optional_context",
        }:
            normalized_role = "substantive"

        roles[normalized_category] = normalized_role

    return roles


def _append_unique(items: List[str], value: str) -> None:
    if value and value not in items:
        items.append(value)

def normalize_evidence_category(value: Any) -> str | None:
    """
    Normalize an evidence category string.

    Evidence categories are plugin-owned open strings, not centrally enumerated.
    """
    if not isinstance(value, str):
        return None

    text = value.strip().lower()

    if not text:
        return None

    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        return None

    return text


def normalize_evidence_categories(categories: Any) -> List[str]:
    if not categories or not isinstance(categories, list):
        return []

    normalized: List[str] = []

    for item in categories:
        value = normalize_evidence_category(item)

        if not value:
            continue

        if value not in normalized:
            normalized.append(value)

    return normalized


def available_evidence_categories_from_plugins() -> List[str]:
    """
    Scan registered plugins and return normalized evidence categories.

    This avoids a central evidence taxonomy file.
    """
    return normalize_evidence_categories(get_available_evidence_categories())


def normalize_coverage_brief(
    brief: Any,
    *,
    allowed_categories: List[str] | None = None,
    drop_unknown: bool = True,
) -> Dict[str, Any]:
    """
    Normalize an LLM- or user-provided coverage brief.

    Only categories with role='substantive' remain in required_evidence_categories.
    Pre-analysis checks and provenance categories are demoted so they do not
    hard-block final answers.
    """
    if not isinstance(brief, dict):
        return {}

    allowed = set(normalize_evidence_categories(allowed_categories or []))
    category_roles = available_evidence_category_roles_from_plugins()

    raw_required = normalize_evidence_categories(
        brief.get("required_evidence_categories")
    )
    raw_prechecks = normalize_evidence_categories(
        brief.get("pre_analysis_check_categories")
    )
    raw_provenance = normalize_evidence_categories(
        brief.get("provenance_categories")
    )
    raw_optional = normalize_evidence_categories(
        brief.get("optional_evidence_categories")
    )

    if drop_unknown and allowed:
        raw_required = [item for item in raw_required if item in allowed]
        raw_prechecks = [item for item in raw_prechecks if item in allowed]
        raw_provenance = [item for item in raw_provenance if item in allowed]
        raw_optional = [item for item in raw_optional if item in allowed]

    required: List[str] = []
    prechecks: List[str] = list(raw_prechecks)
    provenance: List[str] = list(raw_provenance)
    optional: List[str] = list(raw_optional)

    # Demote non-substantive categories even if the LLM put them in required.
    for category in raw_required:
        role = category_roles.get(category, "substantive")

        if role == "substantive":
            _append_unique(required, category)
        elif role == "pre_analysis_check":
            _append_unique(prechecks, category)
        elif role == "provenance":
            _append_unique(provenance, category)
        else:
            _append_unique(optional, category)

    raw_counts = normalize_required_evidence_counts(
        brief.get("required_evidence_counts"),
        allowed_categories=allowed_categories,
        drop_unknown=drop_unknown,
    )

    required_counts: Dict[str, int] = {}

    for category, count in raw_counts.items():
        role = category_roles.get(category, "substantive")

        if role == "substantive":
            required_counts[category] = count
        elif role == "pre_analysis_check":
            _append_unique(prechecks, category)
        elif role == "provenance":
            _append_unique(provenance, category)
        else:
            _append_unique(optional, category)

    return {
        "analysis_goal": str(brief.get("analysis_goal") or "").strip(),
        "required_evidence_categories": required,
        "required_evidence_counts": required_counts,
        "pre_analysis_check_categories": prechecks,
        "provenance_categories": provenance,
        "optional_evidence_categories": optional,
        "autonomy_level": str(brief.get("autonomy_level") or "").strip(),
        "reasoning_summary": str(brief.get("reasoning_summary") or "").strip(),
    }


def covered_evidence_categories_from_runs(
    analysis_runs: List[Dict[str, Any]] | None,
) -> List[str]:
    """
    Compute evidence coverage from actual analysis runs.
    """
    covered: List[str] = []

    for run in analysis_runs or []:
        if run.get("status") not in {"ok", "warning"}:
            continue

        for category in normalize_evidence_categories(run.get("evidence_categories")):
            if category not in covered:
                covered.append(category)

    return covered


def missing_required_evidence_categories(
    *,
    coverage_brief: Dict[str, Any] | None,
    analysis_runs: List[Dict[str, Any]] | None,
    allowed_categories: List[str] | None = None,
) -> List[str]:
    missing_requirements = missing_required_evidence_requirements(
        coverage_brief=coverage_brief,
        analysis_runs=analysis_runs,
        allowed_categories=allowed_categories,
    )

    categories: List[str] = []

    for item in missing_requirements:
        category = item.get("evidence_category")
        if category and category not in categories:
            categories.append(category)

    return categories


def build_evidence_catalog_from_plugins() -> Dict[str, Any]:
    """
    Compact catalog for LLM coverage-brief generation.

    The LLM can use this to choose required evidence categories from currently
    available plugin capabilities.
    """
    categories = available_evidence_categories_from_plugins()

    return {
        "available_evidence_categories": categories,
        "source": "analysis_tool_plugins",
    }

def normalize_required_evidence_counts(
    value: Any,
    *,
    allowed_categories: List[str] | None = None,
    drop_unknown: bool = True,
) -> Dict[str, int]:
    """
    Normalize required evidence counts.

    Example:
    {"group_comparison": 2, "regression_model": 1}
    """
    if not isinstance(value, dict):
        return {}

    allowed = set(normalize_evidence_categories(allowed_categories or []))
    result: Dict[str, int] = {}

    for raw_key, raw_count in value.items():
        category = normalize_evidence_category(raw_key)

        if not category:
            continue

        if drop_unknown and allowed and category not in allowed:
            continue

        try:
            count = int(raw_count)
        except Exception:
            count = 1

        count = max(1, min(count, 10))
        result[category] = count

    return result


def coverage_count_by_category_from_runs(
    analysis_runs: List[Dict[str, Any]] | None,
) -> Dict[str, int]:
    """
    Count successful/warning analysis runs by evidence category.
    """
    counts: Dict[str, int] = {}

    for run in analysis_runs or []:
        if run.get("status") not in {"ok", "warning"}:
            continue

        categories = normalize_evidence_categories(run.get("evidence_categories"))

        for category in categories:
            counts[category] = counts.get(category, 0) + 1

    return counts


def required_evidence_counts_from_brief(
    coverage_brief: Dict[str, Any] | None,
    *,
    allowed_categories: List[str] | None = None,
) -> Dict[str, int]:
    """
    Convert a coverage brief into category -> required count.

    required_evidence_categories gives the baseline required categories.
    required_evidence_counts can increase the required count for selected categories.

    Example:
    required_evidence_categories = [
        "sql_schema",
        "data_preparation",
        "kpi_summary",
        "group_comparison",
        "regression_model",
        "regression_diagnostics",
    ]

    required_evidence_counts = {
        "group_comparison": 2
    }

    Result:
    {
        "sql_schema": 1,
        "data_preparation": 1,
        "kpi_summary": 1,
        "group_comparison": 2,
        "regression_model": 1,
        "regression_diagnostics": 1,
    }
    """
    brief = normalize_coverage_brief(
        coverage_brief,
        allowed_categories=allowed_categories,
        drop_unknown=True,
    )

    result: Dict[str, int] = {}

    for category in normalize_evidence_categories(
        brief.get("required_evidence_categories")
    ):
        result[category] = max(result.get(category, 0), 1)

    explicit_counts = normalize_required_evidence_counts(
        brief.get("required_evidence_counts"),
        allowed_categories=allowed_categories,
        drop_unknown=True,
    )

    for category, count in explicit_counts.items():
        result[category] = max(result.get(category, 0), count)

    return result


def missing_required_evidence_requirements(
    *,
    coverage_brief: Dict[str, Any] | None,
    analysis_runs: List[Dict[str, Any]] | None,
    allowed_categories: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Return detailed missing coverage requirements.

    Example:
    [
      {
        "evidence_category": "group_comparison",
        "required_count": 2,
        "covered_count": 1,
        "missing_count": 1
      }
    ]
    """
    required_counts = required_evidence_counts_from_brief(
        coverage_brief,
        allowed_categories=allowed_categories,
    )
    covered_counts = coverage_count_by_category_from_runs(analysis_runs)

    missing: List[Dict[str, Any]] = []

    for category, required_count in required_counts.items():
        covered_count = covered_counts.get(category, 0)

        if covered_count < required_count:
            missing.append({
                "evidence_category": category,
                "required_count": required_count,
                "covered_count": covered_count,
                "missing_count": required_count - covered_count,
            })

    return missing