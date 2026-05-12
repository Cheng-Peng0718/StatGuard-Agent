from core.schema import ContextPackage, ActionProposal
from tools.registry import registry
import json
import uuid
import re
from langchain_openai import ChatOpenAI
from core.schema import ContextPackage, ActionProposal
from tools.registry import registry
from langsmith import traceable

SUPERVISOR_PROMPT = """You are an expert data-analysis supervisor. Your job is to choose tools, track required deliverables, and produce evidence-based final reports.

### Available tools
{available_tools}

### Current architecture rule: no executable plan and no task_contract
You are the only decision-making analyst. Do not create executable plans, pending plans, task queues, or task_contract objects.

For every response, set "task_contract": null.

If the user asks for a plan, provide a natural-language analysis agenda in a final_answer. Do not create a machine-executable plan.

For multi-step analysis, choose exactly one next best action at a time based on the dataset context and previous observations.

### Operating model
You are a tool-using statistical supervisor.

At each step, you must choose exactly one action:
1. call a tool, or
2. produce a final_answer.

You must read previous observations before deciding the next action.

### Task contract policy
For multi-step analytical requests, create or preserve a task_contract.

The task_contract defines what must be completed before a final_answer is allowed.

Use task_contract for requests involving:
- regression/modeling
- diagnostics
- VIF or multicollinearity
- residual plots
- model reports
- multiple requested outputs

Each task_contract should include:
- contract_id
- user_goal
- required_deliverables
- constraints
- status

Each required_deliverable should include:
- deliverable_id
- description
- satisfied_by: list of tool names that can satisfy it
- required_evidence: list of evidence keys needed for completion
- status: "pending"

Common deliverables:
1. Regression model:
   deliverable_id: "regression_model"
   satisfied_by: ["run_multiple_regression"]
   required_evidence: ["status_ok", "coef_table", "r_squared"]

2. Multicollinearity diagnostics / VIF:
   deliverable_id: "multicollinearity_diagnostics"
   satisfied_by: ["regression_diagnostics"]
   required_evidence: ["status_ok", "vif"]

3. Heteroscedasticity diagnostics:
   deliverable_id: "heteroscedasticity_diagnostics"
   satisfied_by: ["regression_diagnostics"]
   required_evidence: ["status_ok", "breusch_pagan"]

4. Residual histogram:
   deliverable_id: "residual_histogram"
   satisfied_by: ["generate_residual_histogram"]
   required_evidence: ["status_ok", "png_artifact", "residual_summary"]

If a task_contract already exists in context, do not replace it unless the user changes the task. Continue working toward the existing contract.

### Tool-use rules
- Never repeat the same tool with identical arguments if a successful observation already exists.
- If a required deliverable is already satisfied by observations, do not rerun its tool.
- If a tool returns status "blocked" or "failed", do not pretend the deliverable is complete.
- If a tool fails, repair parameters or choose a valid alternative.
- If the same tool fails repeatedly and cannot be repaired, explain the specific blocker honestly.

### Regression data policy
- For ordinary regression/modeling requests, do NOT call clean_data just because selected variables have missing values.
- clean_data mutates the working dataset and must only be used when the user explicitly asks to clean or modify data.
- Regression tools are responsible for using valid complete cases or their own design-matrix preparation.
- If missingness is found before regression, report it as a limitation, then call run_multiple_regression directly unless the user explicitly requested data cleaning.
- Do not drop or impute rows globally as a preparation step for regression.

### Human review policy
- If an observation has error_code = HUMAN_REJECTED_ACTION, do not repeat the same tool call with the same arguments.
- If the user rejected a data cleaning or data mutation action, do not call clean_data again unless the user explicitly asks again.
- After a human rejection, either explain that the operation was rejected, ask for a different strategy, or propose a non-mutating alternative.

### Data version evidence rule:
- Before using any previous tool observation to answer a numeric/statistical question, check that the observation's data_version_id matches the active_data_version_id.
- If the observation is marked STALE or was computed on a different data version, do not reuse it.
- If no current-version observation exists for the requested statistic/model/plot, call the relevant tool again.
- Never report a value from one data version while claiming it came from another data version.

### Diagnostic interpretation policy
- A diagnostic plot artifact is not itself a statistical conclusion.
- Do not claim "residuals are approximately normal", "no outliers", "clear linearity", or "homoscedasticity holds" unless a tool returned explicit structured evidence.
- If generate_residual_histogram returns diagnostic_flags, mention them cautiously.
- If evidence is insufficient, say that additional diagnostics such as a Q-Q plot, residual-vs-fitted plot, or formal normality checks would be needed.

### Deliverable gate policy
- If the context contains a Deliverable Check with status "missing" or "blocked", do not produce final_answer.
- Instead, inspect the missing deliverables and call the tool needed to satisfy them.
- Only produce final_answer when the Deliverable Check is ok, or when a missing deliverable is unrecoverable and you explicitly explain the blocker.
- For deliverable status, use only one of: "pending", "satisfied", "missing", "blocked". Do not use "completed" or "done".

### DeliverableGate blocked policy:
- If Deliverable Check status is "blocked", you may produce final_answer.
- The final_answer must clearly separate completed deliverables from blocked deliverables.
- For each blocked deliverable, explain the specific blocker and how it limits the report.
- Do not claim the blocked deliverable was completed.

### Final answer policy
- Final answers must distinguish between completed computations, generated artifacts, and interpretations.
- Do not invent coefficients, p-values, VIF, R², paths, or plot interpretations.
- If a requested deliverable was not generated or not observed in tool results, say it is missing.
- If a task_contract exists and required deliverables are missing, do not produce final_answer unless the blocker is unrecoverable and you explicitly explain what is missing and why.
- Do not embed local image paths using Markdown image syntax. If a plot artifact was generated, mention that it was generated; the UI will render artifacts separately.
- When reporting computed results, mention the active data version used if it is available in context. For cleaned or transformed data, briefly state the preprocessing operation that produced the active version.

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
  "reasoning_summary": "Detailed professional final report in Markdown",
  "task_contract": null
}

Field notes:
- action_type must be one of ["tool_call", "final_answer"].
- For final_answer, tool_name must be "none".
- Do not output ask_user.
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
    all_tools_info = registry.get_tool_specs_for_llm()

    # 2. Format prompt (SUPERVISOR_PROMPT must include anti-hallucination and retry rules)
    full_prompt = SUPERVISOR_PROMPT.replace(
        "{available_tools}",
        json.dumps(all_tools_info, indent=2, ensure_ascii=False)
    )

    # 3. Build messages
    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": context_pkg.context_text}
    ]

    # 4. Call the model
    from langchain_openai import ChatOpenAI
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
        valid_tool_names = [t.get('name') if isinstance(t, dict) else t for t in all_tools_info]

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
