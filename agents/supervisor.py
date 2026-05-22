import json
import re
import uuid

from langchain_openai import ChatOpenAI
from langsmith import traceable

from core.schema import ContextPackage, ActionProposal
from core.analysis_tool_plugins.registry import get_tool_specs_for_llm


DEBUG_TOOL_CARDS = False

SUPERVISOR_PROMPT = """You are an expert data-analysis supervisor.

You inspect:
1. the user request,
2. the current data context,
3. previous observations and analysis runs,
4. available tool cards,
5. analysis coverage feedback,

then choose exactly ONE next action.

### Available tool cards
{available_tools}

### Non-negotiable architecture rules
You are not a planner.
Do not create executable plans, pending plans, task queues, or task_contract objects.
Always set task_contract to null.
For multi-step analysis, choose one best next action only.

Allowed action_type values:
- tool_call
- final_answer

Do not output ask_user. If clarification is needed, use final_answer and ask the question.

### Decision priority
Follow this order:

1. If CONTINUE_ANALYSIS_RECOMMENDED: true appears in context:
   - final_answer is NOT allowed while missing_evidence_categories is non-empty,
     unless every missing category has no candidate tool.
   - Inspect candidate_tools_for_missing_evidence first.
   - Choose exactly one listed candidate tool that covers one missing evidence category.
   - If the chosen tool has no required arguments, call it with empty arguments: {}.
   - Do not repeat final_answer while evidence is still missing.
   - Do not repeat a successful identical tool call unless the missing evidence count still requires another run.

2. If there is no active dataset but the user provided a SQL database path:
   - Use an appropriate SQL tool.
   - If analysis requires a DataFrame dataset, materialize an analysis-ready dataset first.

3. If an active DataFrame dataset exists:
   - Prefer DataFrame tools for statistical summaries, modeling, diagnostics, plots, and reports.

4. If required information is missing and cannot be inferred:
   - Use final_answer to ask for the missing information.

5. If all requested evidence is already covered:
   - Use final_answer.

### Tool selection rules
Use the tool cards as the source of truth.

Each tool card may include:
- description
- usage_guidance
- use_when
- do_not_use_when
- requires_data_source
- produces_active_dataset
- argument_schema
- examples
- evidence_categories

Choose a tool only when:
- its evidence_categories or usage_guidance match the current need,
- its required arguments can be supplied from the user request, context, observations, or active dataset,
- its do_not_use_when conditions do not apply.

Do not invent tools.
Do not invent placeholder arguments such as:
- your_database_path_here
- path_to_database
- db_path

### Observation and retry rules
Read previous observations before acting.

Reuse previous successful observations if they answer the current need and match the current active_data_version_id.

Do not repeat successful tools with identical arguments unless:
- the data version changed,
- the user explicitly requested rerun,
- the coverage brief requires another distinct run of the same evidence category.

If a tool was blocked or failed:
- read the error,
- revise the next action,
- do not repeat the same failed call,
- if recovery is impossible, explain the blocker in final_answer.

### Data version rules
For numeric/statistical answers, use only results from the current active_data_version_id.

If the active data version changed, recompute statistics, models, diagnostics, and plots before reporting current numeric conclusions.

### Statistical honesty
Never invent:
- coefficients
- p-values
- R²
- VIF
- table values
- file paths
- plot interpretations

Do not claim causality from observational data.
Mention assumptions, limitations, or blockers when relevant.

### SQL and inferential analysis rule
For inferential statistics or modeling, do not materialize one row per group.

Use observation-level data:
- customer-level
- order-level
- subject-level
- patient-level
- transaction-level
- experimental-unit-level

Example:
To test whether total_revenue differs by region, materialize one row per customer or order with both region and total_revenue, not only region-level totals.

### Final answer rules
Use final_answer only when:
- the requested evidence is complete, or
- the task is impossible or blocked, or
- required information is missing.

Final answers must distinguish:
- completed computations,
- generated artifacts,
- blockers or failed tools,
- interpretations,
- limitations,
- recommended next steps.

Mention the active data version when reporting computed results.

### Statistical claims (MANDATORY for final_answer)
When the context lists "Available statistical claims", your final_answer
reasoning_summary MUST express every statistical assertion as a [CLAIM:<id>]
reference using those IDs. Never write p-values, significance verdicts, or
effect-size numbers as your own prose — reference the claim and the system
will substitute the verified wording. Narrative prose around the claims is
encouraged.

### Output format
Output strictly valid JSON only.

For tool_call:
{
  "action_id": "act_01",
  "action_type": "tool_call",
  "tool_name": "concrete_tool_name",
  "arguments": { "param": "value" },
  "reasoning_summary": "Brief rationale",
  "task_contract": null
}

For final_answer:
{
  "action_id": "act_02",
  "action_type": "final_answer",
  "tool_name": "none",
  "arguments": {},
  "reasoning_summary": "Professional Markdown answer",
  "task_contract": null
}

Validation:
- action_type must be one of ["tool_call", "final_answer"].
- For final_answer, tool_name must be "none".
- task_contract must always be null.
"""

def normalize_supervisor_payload(data: dict) -> dict:
    """
    Normalize LLM JSON before Pydantic validation.

    The LLM may use natural-language status values like:
    - completed
    - complete
    - done

    Internally, TaskContract uses:
    - pending
    - satisfied
    - missing
    - blocked
    """
    if not isinstance(data, dict):
        return data

    normalized = dict(data)

    contract = normalized.get("task_contract")

    if isinstance(contract, dict):
        contract = dict(contract)

        deliverables = contract.get("required_deliverables", [])

        if isinstance(deliverables, list):
            normalized_deliverables = []

            for d in deliverables:
                if not isinstance(d, dict):
                    normalized_deliverables.append(d)
                    continue

                d = dict(d)

                if isinstance(d.get("satisfied_by"), str):
                    d["satisfied_by"] = [d["satisfied_by"]]

                if isinstance(d.get("required_evidence"), str):
                    d["required_evidence"] = [d["required_evidence"]]

                status = d.get("status", "pending")

                if isinstance(status, str):
                    s = status.strip().lower()

                    status_map = {
                        "completed": "satisfied",
                        "complete": "satisfied",
                        "done": "satisfied",
                        "finished": "satisfied",
                        "success": "satisfied",
                        "successful": "satisfied",
                        "ok": "satisfied",

                        "not_started": "pending",
                        "not started": "pending",
                        "todo": "pending",
                        "to_do": "pending",

                        "failed": "blocked",
                        "error": "blocked",
                        "failure": "blocked",
                    }

                    d["status"] = status_map.get(s, s)

                if d.get("status") not in {"pending", "satisfied", "missing", "blocked"}:
                    d["status"] = "pending"

                normalized_deliverables.append(d)

            contract["required_deliverables"] = normalized_deliverables

        # Normalize contract-level status too.
        contract_status = contract.get("status", "active")
        if isinstance(contract_status, str):
            cs = contract_status.strip().lower()
            contract_status_map = {
                "completed": "satisfied",
                "complete": "satisfied",
                "done": "satisfied",
                "finished": "satisfied",
                "ok": "satisfied",
            }
            contract["status"] = contract_status_map.get(cs, cs)

        if contract.get("status") not in {"active", "satisfied", "blocked"}:
            contract["status"] = "active"

        normalized["task_contract"] = contract

    return normalized

# LangSmith tracing decorator captures inputs and outputs
@traceable(run_type="llm", name="Supervisor_Reasoning")
def call_supervisor(context_pkg: ContextPackage) -> ActionProposal:
    # 1. Collect all tools
    all_tools_info = get_tool_specs_for_llm()

    # 2. Format prompt with available tool cards.
    full_prompt = SUPERVISOR_PROMPT.replace(
        "{available_tools}",
        json.dumps(all_tools_info, indent=2, ensure_ascii=False)
    )

    if DEBUG_TOOL_CARDS:
        print("\n" + "=" * 40)
        print("[TOOLS VISIBLE TO SUPERVISOR]")
        for name, spec in all_tools_info.items():
            print(
                f"- {name}: "
                f"data_source={spec.get('requires_data_source')}, "
                f"produces_active_dataset={spec.get('produces_active_dataset')}"
            )
        print("=" * 40 + "\n")

    # 3. Build messages
    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": context_pkg.context_text}
    ]

    # 4. Call the model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    response = llm.invoke(messages)
    content = response.content.strip()

    # 5. Extract JSON (models often wrap JSON in prose)
    try:
        json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
    except Exception:
        pass  # let json.loads handle raw content on failure

    # 6. Parse and repair action proposal
    try:
        data = json.loads(content)

        if "action_id" not in data or not data["action_id"]:
            data["action_id"] = f"act_{uuid.uuid4().hex[:8]}"

        valid_types = ["tool_call", "final_answer"]
        raw_type = data.get("action_type")
        valid_tool_names = list(all_tools_info.keys())

        if raw_type not in valid_types and raw_type in valid_tool_names:
            data["tool_name"] = raw_type
            data["action_type"] = "tool_call"
            print(f"[repair]: normalized action_type to 'tool_call'")

        if data.get("action_type") == "tool_call":
            if not data.get("tool_name"):
                raise ValueError("LLM did not provide tool_name")
            if "arguments" not in data:
                data["arguments"] = {}

        data = normalize_supervisor_payload(data)

        # Stabilization mode:
        # task_contract / deliverable hard-gating is disabled for now.
        # The Supervisor remains a one-action-at-a-time analyst.
        data["task_contract"] = None

        return ActionProposal(**data)
    except Exception as e:
        print(f"❌ [parse failed]! type: {type(e).__name__}")
        print(f"LLM raw output -> \n{response.content}")
        raise e
