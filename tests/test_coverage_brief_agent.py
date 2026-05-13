from agents.coverage_brief import _extract_json
from core.analysis_coverage import normalize_coverage_brief
from core.analysis_tool_plugins.registry import get_available_evidence_categories


def test_extract_json_from_plain_json():
    payload = _extract_json('{"analysis_goal": "x", "required_evidence_categories": ["kpi_summary"]}')
    assert payload["analysis_goal"] == "x"
    assert payload["required_evidence_categories"] == ["kpi_summary"]


def test_extract_json_from_markdown_fence():
    payload = _extract_json(
        """
        ```json
        {
          "analysis_goal": "x",
          "required_evidence_categories": ["kpi_summary"]
        }
        ```
        """
    )
    assert payload["analysis_goal"] == "x"


def test_normalized_coverage_brief_supports_required_counts():
    allowed = get_available_evidence_categories()

    brief = normalize_coverage_brief(
        {
            "analysis_goal": "end_to_end",
            "required_evidence_categories": [
                "kpi_summary",
                "group_comparison",
                "regression_model",
            ],
            "required_evidence_counts": {
                "kpi_summary": 1,
                "group_comparison": 2,
                "regression_model": 1,
                "made_up": 5,
            },
            "optional_evidence_categories": [],
            "autonomy_level": "continue_until_covered",
            "reasoning_summary": "Need KPI, two group comparisons, and regression.",
        },
        allowed_categories=allowed,
        drop_unknown=True,
    )

    assert brief["required_evidence_counts"]["kpi_summary"] == 1
    assert brief["required_evidence_counts"]["group_comparison"] == 2
    assert brief["required_evidence_counts"]["regression_model"] == 1
    assert "made_up" not in brief["required_evidence_counts"]