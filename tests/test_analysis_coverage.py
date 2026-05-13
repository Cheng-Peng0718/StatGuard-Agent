from core.analysis_coverage import (
    available_evidence_categories_from_plugins,
    build_evidence_catalog_from_plugins,
    covered_evidence_categories_from_runs,
    missing_required_evidence_categories,
    normalize_coverage_brief,
)


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