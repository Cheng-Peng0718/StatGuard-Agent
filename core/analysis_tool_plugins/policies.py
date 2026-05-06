from __future__ import annotations

from core.analysis_tool_plugins.base import (
    PlanningPolicy,
    VersioningPolicy,
    RepairPolicy,
)


# ----------------------------
# Versioning policies
# ----------------------------

NON_MUTATING_VERSIONING = VersioningPolicy(
    mutates_data=False,
    must_create_child_version=False,
    allowed_to_call_save_df=False,
)


MUTATING_CHILD_VERSIONING = VersioningPolicy(
    mutates_data=True,
    must_create_child_version=True,
    allowed_to_call_save_df=False,
)


# ----------------------------
# Planning policies
# ----------------------------

EDA_READY_PLANNING = PlanningPolicy(
    include_in_capability_map=True,
    ready_without_user_variables=True,
    allow_default_arguments=True,
    requires_variable_contract=False,
    planning_description=(
        "This exploratory analysis can run without user-selected variables."
    ),
)


NEEDS_USER_VARIABLES_PLANNING = PlanningPolicy(
    include_in_capability_map=True,
    ready_without_user_variables=False,
    allow_default_arguments=False,
    requires_variable_contract=True,
    planning_description=(
        "This analysis requires user-selected variables before execution."
    ),
)


MUTATING_REQUIRES_EXPLICIT_INSTRUCTIONS_PLANNING = PlanningPolicy(
    include_in_capability_map=True,
    ready_without_user_variables=False,
    allow_default_arguments=False,
    requires_variable_contract=True,
    planning_description=(
        "This operation mutates the dataset and requires explicit user instructions "
        "and confirmation before execution."
    ),
)


def mutating_requires_choices(*choices: str) -> PlanningPolicy:
    return PlanningPolicy(
        include_in_capability_map=True,
        ready_without_user_variables=False,
        allow_default_arguments=False,
        requires_variable_contract=True,
        required_user_choices=list(choices),
        planning_description=(
            "This operation mutates the dataset and requires explicit user choices "
            "and confirmation before execution."
        ),
    )


def needs_user_choices(*choices: str) -> PlanningPolicy:
    return PlanningPolicy(
        include_in_capability_map=True,
        ready_without_user_variables=False,
        allow_default_arguments=False,
        requires_variable_contract=True,
        required_user_choices=list(choices),
        planning_description=(
            "This analysis requires additional user choices before execution."
        ),
    )


# ----------------------------
# Repair policies
# ----------------------------

DEFAULT_LOW_RISK_REPAIR = RepairPolicy(
    max_attempts=1,
    repairable_error_codes=[],
    non_repairable_error_codes=[
        "INTERNAL_PLUGIN_ERROR",
        "MALFORMED_TOOL_CONTRACT",
    ],
    allow_argument_repair=True,
    allow_method_fallback=False,
    requires_user_for_missing_roles=False,
)


DEFAULT_ANALYSIS_REPAIR = RepairPolicy(
    max_attempts=2,
    repairable_error_codes=[
        "MISSING_VALUES",
        "MISSING_COLUMNS",
        "INVALID_FORMULA",
    ],
    non_repairable_error_codes=[
        "INTERNAL_PLUGIN_ERROR",
        "MALFORMED_TOOL_CONTRACT",
        "DATA_VERSION_NOT_FOUND",
    ],
    allow_argument_repair=True,
    allow_method_fallback=True,
    requires_user_for_missing_roles=True,
)


DEFAULT_MUTATING_REPAIR = RepairPolicy(
    max_attempts=2,
    repairable_error_codes=[
        "INVALID_TOOL_ARGUMENTS",
        "MISSING_COLUMNS",
    ],
    non_repairable_error_codes=[
        "INTERNAL_PLUGIN_ERROR",
        "MALFORMED_TOOL_CONTRACT",
        "DATA_VERSION_UPDATE_INVALID",
    ],
    allow_argument_repair=True,
    allow_method_fallback=False,
    requires_user_for_missing_roles=True,
)

ASSOCIATION_SCREENING_READY_PLANNING = PlanningPolicy(
    include_in_capability_map=True,
    ready_without_user_variables=True,
    allow_default_arguments=True,
    requires_variable_contract=False,
    planning_description=(
        "This association screening analysis can run with default eligible-variable selection."
    ),
)