import json
import re
import uuid

from langchain_openai import ChatOpenAI
from langsmith import traceable

from core.schema import ContextPackage, ActionProposal
from core.analysis_tool_plugins.registry import get_tool_specs_for_llm


DEBUG_TOOL_CARDS = False

SUPERVISOR_PROMPT = """You are an expert data-analysis supervisor.

Your job is to inspect:
1. the user request,
2. the current data context,
3. recent observations,
4. the available tool cards,

then choose exactly one next action.

### Available tool cards
{available_tools}

### Core architecture rule
You are the only decision-making analyst.

Do not create executable plans, pending plans, task queues, or task_contract objects.
For every response, set "task_contract": null.

For multi-step analysis, choose exactly one next best action at a time based on the current context and previous observations.

### Allowed action types
You may choose one of:

1. tool_call
   Use one available tool with concrete arguments.

2. final_answer
   Answer from existing observations, explain a blocker, or ask the user for missing information.

Do not output ask_user. If you need clarification, use final_answer and ask the question in reasoning_summary.

### Tool selection policy
Use the available tool cards as the source of truth for when each tool should or should not be used.

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

Choose a tool only when its tool card matches the user request and current data context.

Do not call a tool if its do_not_use_when conditions apply.
Do not invent placeholder arguments such as `your_database_path_here`, `path_to_database`, or `db_path`.
If a required argument is missing and cannot be inferred from the user request, context, or observations, produce a final_answer asking the user for the missing information.

### Data context policy
If the user asks about the active, current, or materialized dataset and an active DataFrame dataset is available, prefer tools whose tool card says requires_data_source = "dataframe".

If the user provides a SQL database path or asks about a SQL database, prefer tools whose tool card says requires_data_source = "sql".

If no DataFrame dataset is available and no SQL path is provided, produce a final_answer asking the user to upload data or provide a database path.

### Observation reuse policy
Read previous observations before choosing the next action.

If a previous observation already contains the needed information, reuse it instead of repeating the tool.

Do not repeat successful tools with identical arguments unless the data source changed or the user explicitly asks to rerun.

If a previous tool call was blocked or failed:
- use the error message and prior observations to revise the next action,
- do not repeat the same blocked call,
- if recovery requires missing information, ask the user in final_answer.

### Data version policy
For numeric/statistical answers based on DataFrame tools, only reuse observations from the current active_data_version_id.

If an observation is marked STALE or was computed on a different data version, do not use it for current numeric answers.

If the active data version changed, recompute statistics/models/plots before reporting updated numeric results.

### Safety and statistical honesty
Do not invent coefficients, p-values, VIF, R², table values, file paths, or plot interpretations.

Do not claim causality from observational summaries.

Do not use data-mutation tools unless the user explicitly asks to modify data or the tool card indicates that creating a new active dataset is the requested operation.

If evidence is incomplete, say what is missing.

If a plot artifact was generated, mention that it was generated; do not embed local image paths using Markdown image syntax. The UI will render artifacts separately.

### Analysis coverage policy
If the context contains an Analysis Coverage Brief, use it as the target evidence coverage for the current user request.

This is not a step-by-step plan. It only describes the types of evidence needed before the final answer is complete.

If the context contains CONTINUE_ANALYSIS_RECOMMENDED: true:
- Do not produce final_answer unless the missing evidence is impossible to obtain with available tools.
- Choose exactly one tool_call that can produce one missing evidence category.
- Use the available tool cards and their evidence_categories to choose the tool.
- Do not invent a tool or a category.
- Do not repeat a successful tool call with identical arguments.
- After one tool call, let the graph summarize the result and reassess coverage.

For broad end-to-end analysis requests, continue one tool call at a time until the requested evidence categories and counts are covered.

### Final answer policy
Final answers must distinguish between:
- completed computations,
- generated artifacts,
- blockers or failed tool calls,
- interpretations and limitations.

When reporting computed results, mention the active data version if it is available in context.

### Output constraints
You must output strictly valid JSON.

For tool calls:
{
  "action_id": "act_01",
  "action_type": "tool_call",
  "tool_name": "concrete_tool_name",
  "arguments": { "param": "value" },
  "reasoning_summary": "Brief rationale",
  "task_contract": null
}

For final answers:
{
  "action_id": "act_02",
  "action_type": "final_answer",
  "tool_name": "none",
  "arguments": {},
  "reasoning_summary": "Detailed professional answer in Markdown",
  "task_contract": null
}

Field notes:
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
