# UI Adapter Contract

This document defines the stable boundary between the backend graph and any future UI.

The UI must not directly inspect or mutate raw GraphState internals. The UI communicates with the backend through two adapters only:

```python
from core.ui_adapter.events import apply_ui_event_to_state
from core.ui_adapter.snapshot import build_ui_snapshot
```

## Backend to UI

The UI must read backend state through:

```python
snapshot = build_ui_snapshot(state)
```

The returned snapshot is JSON-safe and has schema version:

```text
ui_snapshot_v1
```

The UI may read these top-level snapshot sections:

```text
assistant_response
plan
analysis
data
human_review
runtime
repair
audits
```

## UI to Backend

The UI must send user actions through UI events.

The UI must not directly set fields such as:

```text
user_request
human_review_decision
current_action
current_execution
current_verification
pending_plan
```

Instead, the UI should call:

```python
updates = apply_ui_event_to_state(state, event)
state.update(updates)
```

Supported event types:

```text
user_message
run_plan
approve_human_review
reject_human_review
cancel_plan
select_plan_step
clear_runtime
```

# Examples

## User message

```python
event = {
    "event_type": "user_message",
    "payload": {
        "message": "I want to do analysis to this dataset, what can I do?"
    }
}

updates = apply_ui_event_to_state(state, event)
```

## Run plan

```text
event = {
    "event_type": "run_plan",
    "payload": {}
}

updates = apply_ui_event_to_state(state, event)
```

## Approve human review

```python
event = {
    "event_type": "approve_human_review",
    "payload": {
        "action_hash": "abc123"
    }
}

updates = apply_ui_event_to_state(state, event)
```

## UI must not depend on raw GraphState internals

The future UI must not directly depend on fields like:

```text
final_answer
pending_final_answer
pending_direct_answer
deliverable_gate_allows_final
raw current_action object
raw current_verification object
raw current_execution object
```

The UI should use:

```text
snapshot["assistant_response"]
snapshot["human_review"]
snapshot["runtime"]
snapshot["analysis"]
snapshot["plan"]
snapshot["data"]
snapshot["repair"]
snapshot["audits"]
```

# Old UI freeze

The old `app.py` is considered legacy UI.

Until the new UI is built:

```text
Do not add new business logic to app.py.
Do not add new backend protocols to app.py.
Do not make app.py the source of truth for routing, verification, repair, deliverables, or execution.
Do not make backend modules import app.py or Streamlit.
```

All backend business logic must remain under:

```text
core/
```

The UI layer should remain thin:

```text
read snapshot
render snapshot
send UI events
```

## Future UI architecture

The future UI should follow this loop:

```text
1. User interacts with UI.
2. UI creates UIEvent.
3. UI calls apply_ui_event_to_state().
4. Backend graph nodes run.
5. UI calls build_ui_snapshot().
6. UI renders snapshot.
```

The UI should not know how tools, plans, verification, human review, repair, data versions, or deliverable gates work internally.

