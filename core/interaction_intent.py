from __future__ import annotations

from enum import Enum


class InteractionIntent(str, Enum):
    ADVISORY = "advisory"
    PLAN_ONLY = "plan_only"
    EXECUTE_PLAN = "execute_plan"
    DIRECT_TOOL = "direct_tool"
    UNKNOWN = "unknown"


ADVISORY_PATTERNS = [
    "what can i do",
    "what analyses can i do",
    "what analysis can i do",
    "what should i analyze",
    "what can be analyzed",
    "i want to do analysis",
    "i want to analyze this dataset",
    "what are my options",
    "what would you suggest",
]

PLAN_ONLY_PATTERNS = [
    "make a plan",
    "make up a plan",
    "create a plan",
    "give me a plan",
    "tell me the plan",
    "suggest a plan",
    "analysis plan",
    "plan and tell me",
    "propose a plan",
    "draft a plan",
]

EXECUTE_PLAN_PATTERNS = [
    "run the plan",
    "execute the plan",
    "run this plan",
    "execute this plan",
    "go ahead with the plan",
    "start the plan",
]

DIRECT_TOOL_PATTERNS = [
    "run regression",
    "fit regression",
    "run linear regression",
    "run logistic regression",
    "drop rows",
    "clean data",
    "impute",
    "make a plot",
    "draw a plot",
    "create a histogram",
    "correlation matrix",
    "run correlation",
]


def classify_interaction_intent(user_message: str) -> InteractionIntent:
    """
    Lightweight deterministic intent router.

    This is intentionally conservative:
    - Plan/advisory requests must not execute tools.
    - Execute intent must be explicit.
    - Unknown goes to supervisor as before.
    """
    text = (user_message or "").strip().lower()

    if not text:
        return InteractionIntent.UNKNOWN

    # Execution must be explicit and should win over generic "plan" wording.
    if any(pattern in text for pattern in EXECUTE_PLAN_PATTERNS):
        return InteractionIntent.EXECUTE_PLAN

    if any(pattern in text for pattern in PLAN_ONLY_PATTERNS):
        return InteractionIntent.PLAN_ONLY

    if any(pattern in text for pattern in ADVISORY_PATTERNS):
        return InteractionIntent.ADVISORY

    if any(pattern in text for pattern in DIRECT_TOOL_PATTERNS):
        return InteractionIntent.DIRECT_TOOL

    return InteractionIntent.UNKNOWN