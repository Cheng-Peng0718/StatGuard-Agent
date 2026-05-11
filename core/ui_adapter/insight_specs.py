from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class InsightSpec:
    tool_name: str
    display_name: str
    what_was_computed: str
    default_caveats: List[str] = field(default_factory=list)
    recommended_next_steps: List[str] = field(default_factory=list)
    metric_labels: Dict[str, str] = field(default_factory=dict)


DEFAULT_INSIGHT_SPEC = InsightSpec(
    tool_name="unknown",
    display_name="Analysis Step",
    what_was_computed="An analysis step was run using the current analysis state.",
    default_caveats=[
        "Interpret this result in context of the dataset size, missingness, and analysis assumptions."
    ],
    recommended_next_steps=[
        "Review the result and decide whether another analysis step is needed."
    ],
)


INSIGHT_SPECS: Dict[str, InsightSpec] = {
    "get_summary_stats": InsightSpec(
        tool_name="get_summary_stats",
        display_name="Summary Statistics",
        what_was_computed="Summary statistics were computed for the current dataset.",
        default_caveats=[
            "Summary statistics describe the observed data but do not establish relationships or causality."
        ],
        recommended_next_steps=[
            "Review missingness and relationships between numeric variables before modeling."
        ],
        metric_labels={
            "n_rows": "Rows",
            "n_cols": "Columns",
            "n_numeric_columns": "Numeric columns",
        },
    ),

    "missingness_report": InsightSpec(
        tool_name="missingness_report",
        display_name="Missingness Report",
        what_was_computed="Missing values were summarized for the current dataset.",
        default_caveats=[
            "Missingness can affect downstream modeling and interpretation."
        ],
        recommended_next_steps=[
            "If important variables contain missing values, consider a data cleaning step before modeling."
        ],
    ),

    "get_correlation_matrix": InsightSpec(
        tool_name="get_correlation_matrix",
        display_name="Correlation Matrix",
        what_was_computed="Pairwise correlations were computed among numeric variables.",
        default_caveats=[
            "Correlation is a screening signal and does not imply causation."
        ],
        recommended_next_steps=[
            "Use correlations as screening signals, then confirm relationships with an appropriate model."
        ],
    ),

    "run_multiple_regression": InsightSpec(
        tool_name="run_multiple_regression",
        display_name="Multiple Regression",
        what_was_computed="A multiple regression model was fitted or attempted.",
        default_caveats=[
            "Regression interpretation depends on model assumptions, sample size, missingness, and variable coding."
        ],
        recommended_next_steps=[
            "Check regression diagnostics before relying on coefficient interpretation."
        ],
    ),

    "run_anova": InsightSpec(
        tool_name="run_anova",
        display_name="One-way ANOVA",
        what_was_computed="A one-way ANOVA was run or attempted.",
        default_caveats=[
            "ANOVA interpretation depends on group sizes, variance assumptions, and sample size."
        ],
        recommended_next_steps=[
            "Inspect group sizes and consider follow-up comparisons if the group effect is meaningful."
        ],
    ),

    "clean_data": InsightSpec(
        tool_name="clean_data",
        display_name="Data Cleaning",
        what_was_computed="A data cleaning operation was applied to create a new data version.",
        default_caveats=[
            "Cleaning changes the active dataset and should be interpreted with the data audit log."
        ],
        recommended_next_steps=[
            "Continue analysis using the new active data version created by this cleaning step."
        ],
    ),
}


def get_insight_spec(tool_name: str | None) -> InsightSpec:
    if not tool_name:
        return DEFAULT_INSIGHT_SPEC

    return INSIGHT_SPECS.get(tool_name, DEFAULT_INSIGHT_SPEC)