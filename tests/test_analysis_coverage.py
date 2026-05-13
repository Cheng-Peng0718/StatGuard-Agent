from core.analysis_coverage import (
    available_evidence_categories_from_plugins,
    build_evidence_catalog_from_plugins,
    covered_evidence_categories_from_runs,
    missing_required_evidence_categories,
    normalize_coverage_brief,
    coverage_count_by_category_from_runs,
    missing_required_evidence_requirements,

)
from core.analysis_tool_plugins.registry import get_available_evidence_categories

def test_available_evidence_categories_are_scanned_from_plugins():
    categories = available_evidence_categories_from_plugins()

    assert "kpi_summary" in categories
    assert "group_comparison" in categories
    assert "regression_model" in categories


def test_build_evidence_catalog_from_plugins_is_dynamic():
    catalog = build_evidence_catalog_from_plugins()

    assert catalog["source"] == "analysis_tool_plugins"
    assert "available_evidence_categories" in catalog
    assert "regression_model" in catalog["available_evidence_categories"]


def test_normalize_coverage_brief_filters_to_plugin_declared_categories():
    categories = available_evidence_categories_from_plugins()

    brief = normalize_coverage_brief(
        {
            "analysis_goal": "end to end analysis",
            "required_evidence_categories": [
                "KPI Summary",
                "regression-model",
                "made_up_category",
                "KPI Summary",
            ],
            "optional_evidence_categories": ["Visualization", "fake_optional"],
            "autonomy_level": "continue_until_covered",
            "reasoning_summary": "Need KPI and model evidence.",
        },
        allowed_categories=categories,
        drop_unknown=True,
    )

    assert brief["required_evidence_categories"] == [
        "kpi_summary",
        "regression_model",
    ]


def test_covered_evidence_categories_from_runs_uses_successful_runs_only():
    covered = covered_evidence_categories_from_runs([
        {
            "status": "ok",
            "evidence_categories": ["kpi_summary", "dataset_overview"],
        },
        {
            "status": "blocked",
            "evidence_categories": ["group_comparison"],
        },
        {
            "status": "warning",
            "evidence_categories": ["regression_model"],
        },
    ])

    assert covered == [
        "kpi_summary",
        "dataset_overview",
        "regression_model",
    ]


def test_missing_required_evidence_categories_compares_required_to_covered():
    categories = available_evidence_categories_from_plugins()

    missing = missing_required_evidence_categories(
        coverage_brief={
            "required_evidence_categories": [
                "kpi_summary",
                "group_comparison",
                "regression_model",
            ]
        },
        analysis_runs=[
            {
                "status": "ok",
                "evidence_categories": ["kpi_summary"],
            },
            {
                "status": "ok",
                "evidence_categories": ["regression_model"],
            },
        ],
        allowed_categories=categories,
    )

    assert missing == ["group_comparison"]

def test_missing_required_evidence_requirements_support_counts():
    categories = available_evidence_categories_from_plugins()

    missing = missing_required_evidence_requirements(
        coverage_brief={
            "required_evidence_counts": {
                "kpi_summary": 1,
                "group_comparison": 2,
                "regression_model": 1,
            }
        },
        analysis_runs=[
            {
                "status": "ok",
                "evidence_categories": ["kpi_summary"],
            },
            {
                "status": "ok",
                "evidence_categories": ["group_comparison"],
            },
        ],
        allowed_categories=categories,
    )

    assert missing == [
        {
            "evidence_category": "group_comparison",
            "required_count": 2,
            "covered_count": 1,
            "missing_count": 1,
        },
        {
            "evidence_category": "regression_model",
            "required_count": 1,
            "covered_count": 0,
            "missing_count": 1,
        },
    ]

def test_required_counts_merge_with_required_categories():
    categories = available_evidence_categories_from_plugins()

    missing = missing_required_evidence_requirements(
        coverage_brief={
            "required_evidence_categories": [
                "sql_schema",
                "data_preparation",
                "kpi_summary",
                "group_comparison",
                "regression_model",
                "regression_diagnostics",
            ],
            "required_evidence_counts": {
                "group_comparison": 2,
            },
        },
        analysis_runs=[
            {
                "status": "ok",
                "evidence_categories": ["sql_schema"],
            },
            {
                "status": "ok",
                "evidence_categories": ["data_preparation"],
            },
            {
                "status": "ok",
                "evidence_categories": ["kpi_summary"],
            },
            {
                "status": "ok",
                "evidence_categories": ["group_comparison"],
            },
            {
                "status": "ok",
                "evidence_categories": ["group_comparison"],
            },
            {
                "status": "ok",
                "evidence_categories": ["regression_model"],
            },
        ],
        allowed_categories=categories,
    )

    assert missing == [
        {
            "evidence_category": "regression_diagnostics",
            "required_count": 1,
            "covered_count": 0,
            "missing_count": 1,
        }
    ]

def test_normalize_coverage_brief_demotes_precheck_and_provenance_categories():
    allowed = get_available_evidence_categories()

    brief = normalize_coverage_brief(
        {
            "analysis_goal": "end_to_end",
            "required_evidence_categories": [
                "sql_schema",
                "data_quality",
                "data_preparation",
                "kpi_summary",
                "group_comparison",
                "regression_model",
                "regression_diagnostics",
            ],
            "required_evidence_counts": {
                "sql_schema": 1,
                "data_quality": 1,
                "data_preparation": 1,
                "group_comparison": 2,
                "regression_model": 1,
            },
            "optional_evidence_categories": [],
            "autonomy_level": "continue_until_covered",
            "reasoning_summary": "Need end-to-end evidence.",
        },
        allowed_categories=allowed,
        drop_unknown=True,
    )

    assert brief["required_evidence_categories"] == [
        "kpi_summary",
        "group_comparison",
        "regression_model",
        "regression_diagnostics",
    ]

    assert brief["required_evidence_counts"] == {
        "group_comparison": 2,
        "regression_model": 1,
    }

    assert "data_quality" in brief["pre_analysis_check_categories"]
    assert "sql_schema" in brief["provenance_categories"]
    assert "data_preparation" in brief["provenance_categories"]