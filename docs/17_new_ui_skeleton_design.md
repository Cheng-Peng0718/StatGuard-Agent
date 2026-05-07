# New UI Skeleton Design

This document defines the planned structure for the new UI.

The old `app.py` is legacy and must not receive new backend business logic.

The future UI must communicate with the backend only through:

```python
from core.ui_adapter.events import apply_ui_event_to_state
from core.ui_adapter.snapshot import build_ui_snapshot
```

# Proposed UI entrypoint

The new UI should live in a separate file:

```text
ui/app_v2.py
```

Do not rewrite `app.py` in place until the new UI skeleton is stable.

# UI responsibilities

The UI is responsible for:

```text
1. Rendering the current UI snapshot.
2. Capturing user input.
3. Creating UIEvent objects.
4. Applying UI events to backend state.
5. Triggering backend graph/node execution through a thin controller.
6. Displaying assistant responses, plans, analysis results, data versions, human review prompts, audits, and repair status.
```

The UI is not responsible for:

```text
1. Tool selection.
2. Verification.
3. Human-review policy.
4. Data-version logic.
5. Deliverable-gate logic.
6. Repair decisions.
7. Directly mutating current_action/current_execution/current_verification.
8. Reading raw GraphState internals.
```

# UI session_state contract

The new UI should use only these session_state keys:

```text
backend_state
ui_snapshot
chat_history
uploaded_dataset_info
last_error
```

## backend_state

The raw backend state dictionary.

This is passed to backend nodes and adapters, but the UI should not inspect it directly except through helper functions.

## ui_snapshot

The result of:

```python
ui_snapshot = build_ui_snapshot(backend_state)
```

The UI renders from this object.

## chat_history

A UI-only list of user/assistant messages for display.

The backend source of truth remains `assistant_response`.

## uploaded_dataset_info

UI-only metadata about the uploaded file.

Examples:

```text
filename
n_rows
n_cols
upload_time
```

## last_error

UI-only error message if a UI action fails.

# UI layout

The new UI should have these sections:

```text
1. Chat panel
2. Current assistant response
3. Plan panel
4. Human review panel
5. Analysis results panel
6. Data versions panel
7. Repair/debug panel
8. Audit/debug panel
```

# Chat panel

The chat input creates a UI event:

```python
event = make_user_message_event(user_text)
updates = apply_ui_event_to_state(backend_state, event)
backend_state.update(updates)
```

Then the backend controller runs the appropriate backend flow.

# Run plan button

The run plan button creates:

```python
event = make_run_plan_event()
updates = apply_ui_event_to_state(backend_state, event)
backend_state.update(updates)
```

The UI must not directly call:

```python
backend_state["user_request"] = "run the plan"
```

# Human review buttons

Approve botton:

```python
event = make_approve_human_review_event(
    action_hash=ui_snapshot["human_review"]["action_hash"]
)
```

Reject botton:

```python
event = make_reject_human_review_event(
    action_hash=ui_snapshot["human_review"]["action_hash"],
    reason=user_reason
)
```

The UI must not directly set:

```python
The UI must not directly set:
```

# Rendering assistant response

The UI reads:

```python
snapshot["assistant_response"]
```

Expected shape:

```text
response_type
content
source_node
metadata
```

# Rendering plan

The UI reads:

```python
snapshot["plan"]["pending_plan"]
snapshot["plan"]["plan_status"]
snapshot["plan"]["plan_execution_status"]
```

# Rendering analysis results

The UI reads:

```python
snapshot["analysis"]["analysis_runs"]
snapshot["analysis"]["observations"]
```

The UI should not parse raw tool payloads directly unless they are exposed through `analysis_runs`.

# Rendering data versions

The UI reads:

```python
snapshot["data"]["active_data_version_id"]
snapshot["data"]["data_versions"]
snapshot["data"]["data_audit_log"]
```

# Rendering human review

The UI reads:

```python
snapshot["human_review"]
```

If:

```python
snapshot["human_review"]["required"] is True
```

then the UI shows approve/reject controls.

# Rendering repair/debug information

The UI reads:

```python
snapshot["repair"]["decision"]
snapshot["repair"]["proposal"]
snapshot["repair"]["attempts"]
```

This section can be hidden behind an expandable debug panel.

# Rendering audits

The UI reads:

```python
snapshot["audits"]["execution_audit"]
snapshot["audits"]["state_serialization_audit"]
snapshot["audits"]["deliverable_check"]
```

This section can be hidden behind an expandable debug panel.

# Backend controller boundary

The future UI should not manually call many backend nodes inline.

Instead, it should call a thin backend controller function such as:

```python
run_backend_turn(backend_state)
```

That controller is future work.

For now, the UI skeleton should only depend on:

```text
apply_ui_event_to_state
build_ui_snapshot
```

# Forbidden UI behavior

The new UI must not:

```text
1. Import tool plugins directly.
2. Call execute_node directly from arbitrary widget callbacks.
3. Mutate current_action/current_execution/current_verification manually.
4. Inspect final_answer/pending_final_answer/pending_direct_answer.
5. Implement deliverable-gate logic.
6. Implement repair logic.
7. Implement data-version logic.
8. Import legacy app.py.
```

