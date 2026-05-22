from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langsmith import traceable

from core.analysis_coverage import (
    build_evidence_catalog_from_plugins,
    normalize_coverage_brief,
)


COVERAGE_BRIEF_PROMPT = """You are an evidence coverage analyst for a data analysis agent.

Your job is to read:
1. the user request,
2. the current data context,
3. the evidence categories currently available from registered tools,

and produce a compact Analysis Coverage Brief.

This is NOT a workflow plan.
Do NOT create step-by-step plans.
Do NOT choose exact tools.
Do NOT invent evidence categories not listed in available_evidence_categories.

The brief only says what types of evidence should exist before a final answer/report is considered complete.

Available evidence catalog:
{evidence_catalog}

Output strictly valid JSON with this schema:

{
  "analysis_goal": "short_snake_case_goal_name",
  "required_evidence_categories": ["category_from_available_list"],
  "required_evidence_counts": {
    "category_from_available_list": 1
  },
  "pre_analysis_check_categories": ["category_from_available_list"],
  "provenance_categories": ["category_from_available_list"],
  "optional_evidence_categories": ["category_from_available_list"],
  "autonomy_level": "answer_now | continue_until_covered | ask_user",
  "reasoning_summary": "one or two sentences explaining why these evidence categories are needed"
}

Rules:
- For simple schema inspection or data lookup requests, require only the directly relevant evidence.
- For broad end-to-end analysis requests, require enough evidence categories to support the requested final report.
- If the user asks to summarize KPIs, include kpi_summary if available.
- If the user asks to compare groups, segments, regions, treatments, or categories, include group_comparison if available.
- If the user asks to model drivers, predictors, or explain a NUMERIC outcome from other columns, include regression_model if available.
- If the user asks about association/relationship between two columns, choose by column type shown in the context: when BOTH columns are categorical, the appropriate evidence is statistical_inference (a chi-square test), NOT regression_model; when the outcome is numeric, regression_model or group_comparison is appropriate. Do not require regression_model for two categorical variables.
- If the user asks to diagnose/check model reliability/assumptions, include regression_diagnostics if available.
- If the user asks to compare more than one grouping variable, set required_evidence_counts.group_comparison to the number of distinct requested comparisons.
- Use required_evidence_counts to indicate how many successful pieces of evidence of each category are needed.
- Choose autonomy_level = continue_until_covered for broad analysis requests where the agent should keep calling tools until the required evidence is covered.
- Choose autonomy_level = answer_now for simple requests where one answer is enough.
- Choose autonomy_level = ask_user only when the request cannot be interpreted or required inputs are missing.
Evidence category role rules:
- required_evidence_categories should contain only substantive final-answer evidence.
- Do NOT put data_quality, missingness, data_profile, dataset_overview, sql_schema, or data_preparation in required_evidence_categories unless the user explicitly asks for that item as the final deliverable.
- For broad analysis requests, put data_quality or missingness in pre_analysis_check_categories.
- Put sql_schema and data_preparation in provenance_categories when relevant.
- pre_analysis_check_categories and provenance_categories are useful for report quality, but they should not block final_answer if missing.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    content = str(text or "").strip()

    try:
        return json.loads(content)
    except Exception:
        pass

    match = re.search(r"(\{.*\})", content, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        return json.loads(content)

    if "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(content)

    raise ValueError("Could not parse JSON coverage brief from LLM response.")


@traceable(run_type="llm", name="Analysis_Coverage_Brief")
def call_coverage_brief(
    *,
    user_request: str,
    context_text: str = "",
) -> Dict[str, Any]:
    evidence_catalog = build_evidence_catalog_from_plugins()
    available_categories = evidence_catalog.get("available_evidence_categories", [])

    system_prompt = COVERAGE_BRIEF_PROMPT.replace(
        "{evidence_catalog}",
        json.dumps(evidence_catalog, indent=2, ensure_ascii=False),
    )

    user_prompt = (
        "User request:\n"
        f"{user_request}\n\n"
        "Current context:\n"
        f"{context_text[:6000]}\n"
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    parsed = _extract_json(response.content)

    return normalize_coverage_brief(
        parsed,
        allowed_categories=available_categories,
        drop_unknown=True,
    )