"""
One-way ANOVA plugin.

This module is now a thin wrapper that delegates to the higher-quality
statistical_group_comparison plugin so that both `run_anova` and
`statistical_group_comparison` produce consistent, rigorous output:
Levene's test for variance homogeneity, automatic switch between classic
ANOVA (equal variances) and Welch's/Alexander-Govern ANOVA (unequal variances),
FWER-controlled post-hoc pairwise tests (Tukey HSD or Games-Howell),
Shapiro-Wilk per-group normality, eta-squared and omega-squared effect sizes
with magnitude interpretation, and assumption tables.

The legacy `run_anova` tool name is preserved for backwards compatibility with
existing supervisor selections, but the execute and extract functions now
defer to statistical_group_comparison.
"""

from typing import Any, Dict, Tuple

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.group_comparison_guardrails import (
    evaluate_group_comparison_guardrails,
)

from core.analysis_tool_plugins.plugins.statistical_group_comparison import (
    execute_statistical_group_comparison,
    extract_statistical_group_comparison,
    STATISTICAL_GROUP_COMPARISON_DISPLAY,
)


def execute_anova(context) -> Dict[str, Any]:
    """
    Delegate to the upgraded statistical_group_comparison execute, then
    translate the error code for backwards compatibility with the existing
    run_anova contract.
    """
    result = execute_statistical_group_comparison(context)

    # Translate error code (singular -> plural) to match the legacy run_anova
    # contract that downstream tests and prompts may rely on.
    if (
        result.get("status") == "blocked"
        and result.get("error_code") == "COLUMN_NOT_FOUND"
    ):
        result["error_code"] = "COLUMNS_NOT_FOUND"

    return result


def extract_anova(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title, summary, metrics, tables, metadata = extract_statistical_group_comparison(
        payload=payload,
        arguments=arguments,
        default_title=default_title,
        default_summary=default_summary,
    )

    # Override title to preserve the run_anova display contract.
    target_col = payload.get("target_col") or arguments.get("target_col")
    group_col = payload.get("group_col") or arguments.get("group_col")

    if target_col and group_col:
        title = f"One-way ANOVA: {target_col} by {group_col}"
    else:
        title = "One-way ANOVA"

    return title, summary, metrics, tables, metadata


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_anova",
    display_name="One-way ANOVA",
    description=(
        "One-way ANOVA with Levene's test for variance homogeneity. "
        "Automatically switches between classic ANOVA (equal variances) and "
        "Welch's/Alexander-Govern ANOVA (unequal variances). When the omnibus "
        "test is significant, FWER-controlled pairwise comparisons follow "
        "(Tukey HSD for equal variances, Games-Howell otherwise). Reports "
        "eta-squared and omega-squared effect sizes, magnitude interpretation, "
        "and Shapiro-Wilk per-group normality."
    ),
    usage_guidance=(
        "Use this when the user explicitly asks for a one-way ANOVA on a numeric "
        "outcome across a categorical group. For a more general two-or-more-group "
        "comparison entry point (which also handles the two-group case via Welch's "
        "t-test), prefer `statistical_group_comparison`."
    ),
    requires_confirmation=False,
    is_inferential=True,
    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "group_col": str,
        },
        optional={
            "alpha": float,
        },
        column_args=[
            "target_col",
            "group_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_anova,
    extractor=extract_anova,
    guardrail_evaluators=[
        evaluate_group_comparison_guardrails,
    ],
    display_config=STATISTICAL_GROUP_COMPARISON_DISPLAY,
))