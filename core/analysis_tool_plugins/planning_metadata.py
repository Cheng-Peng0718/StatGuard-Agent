from __future__ import annotations

REGRESSION_TASK_ARGUMENT_BINDINGS = [
    {
        "task_field": "target_variables",
        "index": 0,
        "argument": "target_col",
        "required_choice": "target_col",
    },
    {
        "task_field": "predictor_variables",
        "argument": "feature_cols",
        "required_choice": "feature_cols",
    },
]


CLEAN_DATA_TASK_ARGUMENT_BINDINGS = [
    {
        "task_field": "target_variables",
        "argument": "columns",
        "required_choice": "columns",
    },
]


TOOL_PLANNING_METADATA = {
    "inspect_dataset": {
        "supported_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning", "eda"],
        "planning_tags": ["overview", "schema", "eda"],
        "default_plan_purpose": "Inspect dataset shape, columns, and basic schema.",
        "expected_deliverables": ["dataset_overview"],
        "plan_order": 10,
    },
    "missingness_report": {
        "supported_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning", "eda"],
        "planning_tags": ["overview", "missingness", "data_quality", "eda"],
        "default_plan_purpose": "Assess missing values before recommending analyses.",
        "expected_deliverables": ["missingness_assessment"],
        "plan_order": 20,
    },
    "get_summary_stats": {
        "supported_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning", "eda"],
        "planning_tags": ["overview", "descriptive_statistics", "numeric_summary", "eda"],
        "default_plan_purpose": "Summarize numeric variables with descriptive statistics.",
        "expected_deliverables": ["descriptive_statistics"],
        "plan_order": 30,
    },
    "summarize_columns": {
        "supported_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning", "eda"],
        "planning_tags": ["overview", "column_summary", "eda"],
        "default_plan_purpose": "Summarize column-level distributions and types.",
        "expected_deliverables": ["column_summary"],
        "plan_order": 40,
    },
    "get_correlation_matrix": {
        "supported_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning", "association_analysis", "eda"],
        "planning_tags": ["association", "correlation", "screening", "eda"],
        "default_plan_purpose": "Screen numeric associations only when enough numeric variables exist.",
        "expected_deliverables": ["correlation_screening"],
        "plan_order": 50,
    },
    "run_multiple_regression": {
        "supported_goal_types": ["regression_modeling"],
        "not_recommended_for_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning"],
        "planning_tags": ["regression", "modeling", "inferential"],
        "default_plan_purpose": "Fit the requested regression model.",
        "expected_deliverables": ["regression_model"],
        "task_argument_bindings": REGRESSION_TASK_ARGUMENT_BINDINGS,
        "plan_order": 10,
    },
    "regression_diagnostics": {
        "supported_goal_types": ["regression_modeling"],
        "not_recommended_for_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning"],
        "planning_tags": ["regression", "diagnostics", "model_checking"],
        "default_plan_purpose": "Check model diagnostics after the regression fit.",
        "expected_deliverables": ["regression_diagnostics"],
        "task_argument_bindings": REGRESSION_TASK_ARGUMENT_BINDINGS,
        "plan_order": 20,
    },
    "generate_residual_histogram": {
        "supported_goal_types": ["regression_modeling", "visualization"],
        "not_recommended_for_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning"],
        "planning_tags": ["regression", "diagnostics", "visualization", "residuals"],
        "default_plan_purpose": "Generate residual distribution evidence after fitting the model.",
        "expected_deliverables": ["residual_distribution"],
        "task_argument_bindings": REGRESSION_TASK_ARGUMENT_BINDINGS,
        "plan_order": 30,
    },
    "run_correlation_test": {
        "supported_goal_types": ["association_analysis"],
        "planning_tags": ["association", "correlation", "inferential"],
        "default_plan_purpose": "Test association between selected variables.",
        "expected_deliverables": ["association_test"],
        "plan_order": 10,
    },
    "generate_scatterplot": {
        "supported_goal_types": ["association_analysis", "visualization"],
        "planning_tags": ["association", "visualization", "scatterplot"],
        "default_plan_purpose": "Visualize the selected variables.",
        "expected_deliverables": ["scatterplot"],
        "plan_order": 10,
    },
    "clean_data": {
        "supported_goal_types": ["data_cleaning"],
        "not_recommended_for_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning"],
        "planning_tags": ["data_cleaning", "mutation", "requires_confirmation"],
        "default_plan_purpose": "Prepare a data modification proposal that requires user confirmation.",
        "expected_deliverables": ["cleaned_dataset_version"],
        "task_argument_bindings": CLEAN_DATA_TASK_ARGUMENT_BINDINGS,
        "required_planning_choices": [
          "action_type",
          "strategy",
        ],
        "plan_order": 10,
    },
    "run_anova": {
        "supported_goal_types": ["group_comparison"],
        "not_recommended_for_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning"],
        "planning_tags": ["group_comparison", "anova", "inferential"],
        "default_plan_purpose": "Compare numeric outcomes across groups with one-way ANOVA.",
        "expected_deliverables": ["group_comparison_test"],
        "plan_order": 10,
    },
    "run_chi_square": {
        "supported_goal_types": ["categorical_association", "association_analysis"],
        "not_recommended_for_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning"],
        "planning_tags": ["categorical_association", "chi_square", "inferential"],
        "default_plan_purpose": "Test association between categorical variables.",
        "expected_deliverables": ["categorical_association_test"],
        "plan_order": 10,
    },
    "run_independent_t_test": {
        "supported_goal_types": ["group_comparison"],
        "not_recommended_for_goal_types": ["dataset_overview", "analysis_recommendation", "analysis_planning"],
        "planning_tags": ["group_comparison", "t_test", "inferential"],
        "default_plan_purpose": "Compare a numeric outcome between two independent groups.",
        "expected_deliverables": ["group_comparison_test"],
        "plan_order": 10,
    },
}
