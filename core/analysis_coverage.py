from __future__ import annotations

import re
from typing import Any, Dict, List

from core.analysis_tool_plugins.registry import get_available_evidence_categories


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

    If allowed_categories is provided, required/optional categories are filtered
    to categories currently declared by registered plugins.
    """
    if not isinstance(brief, dict):
        return {}

    allowed = set(normalize_evidence_categories(allowed_categories or []))

    required = normalize_evidence_categories(
        brief.get("required_evidence_categories")
    )
    optional = normalize_evidence_categories(
        brief.get("optional_evidence_categories")
    )

    if drop_unknown and allowed:
        required = [item for item in required if item in allowed]
        optional = [item for item in optional if item in allowed]

    return {
        "analysis_goal": str(brief.get("analysis_goal") or "").strip(),
        "required_evidence_categories": required,
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
    """
    Compare required evidence from a coverage brief against evidence produced by analysis runs.
    """
    brief = normalize_coverage_brief(
        coverage_brief,
        allowed_categories=allowed_categories,
        drop_unknown=True,
    )

    required = brief.get("required_evidence_categories", [])
    covered = set(covered_evidence_categories_from_runs(analysis_runs))

    return [category for category in required if category not in covered]


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